// waifu2x implemented with ncnn library

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <string>
#include <sstream>
#include <clocale>

// image decoder and encoder with libjpeg and libpng
#include "jpeg_image.h"
#include "png_image.h"
#include "webp_image.h"

#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "waifu2x.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stdout, "Usage: waifu2x-ncnn-vulkan -i infile -o outfile [options]...\n\n");
    fprintf(stdout, "  -h                   show this help\n");
    fprintf(stdout, "  -v                   verbose output\n");
    fprintf(stdout, "  -i input-path        input image path (jpg/png/webp)\n");
    fprintf(stdout, "  -o output-path       output image path (jpg/png/webp)\n");
    fprintf(stdout, "  -n noise-level       denoise level (-1/0/1/2/3, default=0)\n");
    fprintf(stdout, "  -s scale             upscale ratio (1/2/4/8/16/32, default=2)\n");
    fprintf(stdout, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stdout, "  -m model-path        waifu2x model path (default=models-cunet)\n");
    fprintf(stdout, "  -g gpu-id            gpu device to use (-1=cpu, default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stdout, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stdout, "  -x                   enable tta mode\n");
    fprintf(stdout, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
    fprintf(stdout, "  -D pipe-file         daemon mode, read request arguments from pipe file\n");
}

class Options
{
public:
    path_t inputpath;
    path_t outputpath;
    int noise = 0;
    int scale = 2;
    std::vector<int> tilesize;
    path_t model = PATHSTR("models-cunet");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    path_t format = PATHSTR("png");
    path_t daemon_pipe;
};

static bool is_daemon_conflicting_option(int opt)
{
    return opt == 'n' || opt == 's' || opt == 't' || opt == 'm' || opt == 'g' || opt == 'j' || opt == 'x' || opt == 'v';
}

static int parse_options(int argc, char** argv, Options& opt, bool daemon_request)
{
    optind = 1;
    int c;
    while ((c = getopt(argc, argv, "i:o:n:s:t:m:g:j:f:D:vxh")) != -1)
    {
        if (daemon_request && is_daemon_conflicting_option(c))
        {
            fprintf(stderr, "warning: ignoring -%c in daemon request\n", c);
            continue;
        }

        switch (c)
        {
        case 'i':
            opt.inputpath = optarg;
            break;
        case 'o':
            opt.outputpath = optarg;
            break;
        case 'n':
            opt.noise = atoi(optarg);
            break;
        case 's':
            opt.scale = atoi(optarg);
            break;
        case 't':
            opt.tilesize = parse_optarg_int_array(optarg);
            break;
        case 'm':
            opt.model = optarg;
            break;
        case 'g':
            opt.gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &opt.jobs_load, &opt.jobs_save);
            opt.jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            opt.format = optarg;
            break;
        case 'D':
            if (daemon_request)
            {
                fprintf(stderr, "warning: ignoring -D in daemon request\n");
            }
            else
            {
                opt.daemon_pipe = optarg;
            }
            break;
        case 'v':
            opt.verbose = 1;
            break;
        case 'x':
            opt.tta_mode = 1;
            break;
        case 'h':
        default:
            return -1;
        }
    }

    return 0;
}


class Task
{
public:
    int id;
    int scale;

    path_t inpath;
    path_t outpath;

    ncnn::Mat inimage;
    ncnn::Mat outimage;
};

class TaskQueue
{
public:
    TaskQueue()
    {
    }

    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 8) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;
ncnn::Mutex g_done_lock;
ncnn::ConditionVariable g_done_condition;
int g_done_count = 0;

class LoadThreadParams
{
public:
    int scale;
    int jobs_load;

    // session data
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    const int count = ltp->input_files.size();
    const int scale = ltp->scale;

    #pragma omp parallel for schedule(static,1) num_threads(ltp->jobs_load)
    for (int i=0; i<count; i++)
    {
        const path_t& imagepath = ltp->input_files[i];

        unsigned char* pixeldata = 0;
        int w;
        int h;
        int c;

#if _WIN32
        FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
        FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
        if (fp)
        {
            // read whole file
            unsigned char* filedata = 0;
            int length = 0;
            {
                fseek(fp, 0, SEEK_END);
                length = ftell(fp);
                rewind(fp);
                filedata = (unsigned char*)malloc(length);
                if (filedata)
                {
                    fread(filedata, 1, length, fp);
                }
                fclose(fp);
            }

            if (filedata)
            {
                pixeldata = webp_load(filedata, length, &w, &h, &c);
                if (!pixeldata)
                {
                    // not webp, try jpg png etc.
#if _WIN32
                    pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else // _WIN32
                    pixeldata = jpeg_load(filedata, length, &w, &h, &c);
                    if (!pixeldata)
                    {
                        pixeldata = png_load(filedata, length, &w, &h, &c);
                    }
#endif // _WIN32
                }

                free(filedata);
            }
        }
        if (pixeldata)
        {
            Task v;
            v.id = i;
            v.scale = scale;
            v.inpath = imagepath;
            v.outpath = ltp->output_files[i];

            v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);

            path_t ext = get_file_extension(v.outpath);
            if (c == 4 && (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")))
            {
                path_t output_filename2 = ltp->output_files[i] + PATHSTR(".png");
                v.outpath = output_filename2;
#if _WIN32
                fwprintf(stderr, L"image %ls has alpha channel ! %ls will output %ls\n", imagepath.c_str(), imagepath.c_str(), output_filename2.c_str());
#else // _WIN32
                fprintf(stderr, "image %s has alpha channel ! %s will output %s\n", imagepath.c_str(), imagepath.c_str(), output_filename2.c_str());
#endif // _WIN32
            }

            toproc.put(v);
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"decode image %ls failed\n", imagepath.c_str());
#else // _WIN32
            fprintf(stderr, "decode image %s failed\n", imagepath.c_str());
#endif // _WIN32

            g_done_lock.lock();
            g_done_count++;
            g_done_lock.unlock();
            g_done_condition.signal();
        }
    }

    return 0;
}

class ProcThreadParams
{
public:
    const Waifu2x* waifu2x;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const Waifu2x* waifu2x = ptp->waifu2x;

    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233)
            break;

        const int scale = v.scale;
        if (scale == 1)
        {
            v.outimage = ncnn::Mat(v.inimage.w, v.inimage.h, (size_t)v.inimage.elemsize, (int)v.inimage.elemsize);
            waifu2x->process(v.inimage, v.outimage);

            tosave.put(v);
            continue;
        }

        int scale_run_count = 0;
        if (scale == 2)
        {
            scale_run_count = 1;
        }
        if (scale == 4)
        {
            scale_run_count = 2;
        }
        if (scale == 8)
        {
            scale_run_count = 3;
        }
        if (scale == 16)
        {
            scale_run_count = 4;
        }
        if (scale == 32)
        {
            scale_run_count = 5;
        }

        v.outimage = ncnn::Mat(v.inimage.w * 2, v.inimage.h * 2, (size_t)v.inimage.elemsize, (int)v.inimage.elemsize);
        waifu2x->process(v.inimage, v.outimage);

        for (int i = 1; i < scale_run_count; i++)
        {
            ncnn::Mat tmp = v.outimage;
            v.outimage = ncnn::Mat(tmp.w * 2, tmp.h * 2, (size_t)v.inimage.elemsize, (int)v.inimage.elemsize);
            waifu2x->process(tmp, v.outimage);
        }

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams
{
public:
    int verbose;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;

    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233)
            break;

        // free input pixel data
        {
            unsigned char* pixeldata = (unsigned char*)v.inimage.data;
            free(pixeldata);
        }

        int success = 0;

        path_t ext = get_file_extension(v.outpath);

        if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            success = webp_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, (const unsigned char*)v.outimage.data);
        }
        else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
#if _WIN32
            success = wic_encode_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
            success = png_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, (const unsigned char*)v.outimage.data);
#endif
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
#if _WIN32
            success = wic_encode_jpeg_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
            success = jpeg_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, (const unsigned char*)v.outimage.data);
#endif
        }
        if (success)
        {
            if (verbose)
            {
#if _WIN32
                fwprintf(stdout, L"%ls -> %ls done\n", v.inpath.c_str(), v.outpath.c_str());
#else
                fprintf(stdout, "%s -> %s done\n", v.inpath.c_str(), v.outpath.c_str());
#endif
            }
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"encode image %ls failed\n", v.outpath.c_str());
#else
            fprintf(stderr, "encode image %s failed\n", v.outpath.c_str());
#endif
        }

        g_done_lock.lock();
        g_done_count++;
        g_done_lock.unlock();
        g_done_condition.signal();
    }

    return 0;
}


int main(int argc, char** argv)
{
    Options opt;

    if (parse_options(argc, argv, opt, false) != 0)
    {
        print_usage();
        return -1;
    }

    path_t inputpath = opt.inputpath;
    path_t outputpath = opt.outputpath;
    int noise = opt.noise;
    int scale = opt.scale;
    std::vector<int> tilesize = opt.tilesize;
    path_t model = opt.model;
    std::vector<int> gpuid = opt.gpuid;
    int jobs_load = opt.jobs_load;
    std::vector<int> jobs_proc = opt.jobs_proc;
    int jobs_save = opt.jobs_save;
    int verbose = opt.verbose;
    int tta_mode = opt.tta_mode;
    path_t format = opt.format;

    if (opt.daemon_pipe.empty() && (inputpath.empty() || outputpath.empty()))
    {
        print_usage();
        return -1;
    }

    if (noise < -1 || noise > 3)
    {
        fprintf(stderr, "invalid noise argument\n");
        return -1;
    }

    if (!(scale == 1 || scale == 2 || scale == 4 || scale == 8 || scale == 16 || scale == 32))
    {
        fprintf(stderr, "invalid scale argument\n");
        return -1;
    }

    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) && !tilesize.empty())
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    for (int i=0; i<(int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "invalid tilesize argument\n");
            return -1;
        }
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }

    if (path_is_directory(inputpath) || path_is_directory(outputpath))
    {
        fprintf(stderr, "directory input/output is not supported\n");
        return -1;
    }

    // guess format from outputpath no matter what format argument specified
    path_t ext = get_file_extension(outputpath);

    if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
    {
        format = PATHSTR("png");
    }
    else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
    {
        format = PATHSTR("webp");
    }
    else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
    {
        format = PATHSTR("jpg");
    }
    else
    {
        fprintf(stderr, "invalid outputpath extension type\n");
        return -1;
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg"))
    {
        fprintf(stderr, "invalid format argument\n");
        return -1;
    }

    int prepadding = 0;

    if (model.find(PATHSTR("models-cunet")) != path_t::npos)
    {
        if (noise == -1)
        {
            prepadding = 18;
        }
        else if (scale == 1)
        {
            prepadding = 28;
        }
        else if (scale == 2 || scale == 4 || scale == 8 || scale == 16 || scale == 32)
        {
            prepadding = 18;
        }
    }
    else if (model.find(PATHSTR("models-upconv_7_anime_style_art_rgb")) != path_t::npos)
    {
        prepadding = 7;
    }
    else if (model.find(PATHSTR("models-upconv_7_photo")) != path_t::npos)
    {
        prepadding = 7;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    char parampath[256];
    char modelpath[256];
    if (noise == -1)
    {
        sprintf(parampath, "%s/scale2.0x_model.param", model.c_str());
        sprintf(modelpath, "%s/scale2.0x_model.bin", model.c_str());
    }
    else if (scale == 1)
    {
        sprintf(parampath, "%s/noise%d_model.param", model.c_str(), noise);
        sprintf(modelpath, "%s/noise%d_model.bin", model.c_str(), noise);
    }
    else if (scale == 2 || scale == 4 || scale == 8 || scale == 16 || scale == 32)
    {
        sprintf(parampath, "%s/noise%d_scale2.0x_model.param", model.c_str(), noise);
        sprintf(modelpath, "%s/noise%d_scale2.0x_model.bin", model.c_str(), noise);
    }

    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < -1 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] == -1)
        {
            jobs_proc[i] = std::min(jobs_proc[i], cpu_count);
            total_jobs_proc += 1;
        }
        else
        {
            total_jobs_proc += jobs_proc[i];
        }
    }

    for (int i=0; i<use_gpu_count; i++)
    {
        if (tilesize[i] != 0)
            continue;

        if (gpuid[i] == -1)
        {
            // cpu only
            tilesize[i] = 400;
            continue;
        }

        uint32_t heap_budget = ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

        // more fine-grained tilesize policy here
        if (model.find(PATHSTR("models-cunet")) != path_t::npos)
        {
            if (heap_budget > 2600)
                tilesize[i] = 400;
            else if (heap_budget > 740)
                tilesize[i] = 200;
            else if (heap_budget > 250)
                tilesize[i] = 100;
            else
                tilesize[i] = 32;
        }
        else if (model.find(PATHSTR("models-upconv_7_anime_style_art_rgb")) != path_t::npos
            || model.find(PATHSTR("models-upconv_7_photo")) != path_t::npos)
        {
            if (heap_budget > 1900)
                tilesize[i] = 400;
            else if (heap_budget > 550)
                tilesize[i] = 200;
            else if (heap_budget > 190)
                tilesize[i] = 100;
            else
                tilesize[i] = 32;
        }
    }

    {
        std::vector<Waifu2x*> waifu2x(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            int num_threads = gpuid[i] == -1 ? jobs_proc[i] : 1;

            waifu2x[i] = new Waifu2x(gpuid[i], tta_mode, num_threads);

            waifu2x[i]->load(paramfullpath, modelfullpath);

            waifu2x[i]->noise = noise;
            waifu2x[i]->scale = (scale >= 2) ? 2 : scale;
            waifu2x[i]->tilesize = tilesize[i];
            waifu2x[i]->prepadding = prepadding;
        }

        // main routine
        {
            // waifu2x proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].waifu2x = waifu2x[i];
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    if (gpuid[i] == -1)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                    else
                    {
                        for (int j=0; j<jobs_proc[i]; j++)
                        {
                            proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                        }
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i] = new ncnn::Thread(save, (void*)&stp);
            }

            auto run_one_request = [&](const path_t& req_input, const path_t& req_output, path_t req_format) -> int
            {
                if (req_input.empty() || req_output.empty())
                {
                    fprintf(stderr, "invalid request, missing input or output\n");
                    return -1;
                }

                if (path_is_directory(req_input) || path_is_directory(req_output))
                {
                    fprintf(stderr, "directory input/output is not supported\n");
                    return -1;
                }

                path_t ext = get_file_extension(req_output);
                if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
                    req_format = PATHSTR("png");
                else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
                    req_format = PATHSTR("webp");
                else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
                    req_format = PATHSTR("jpg");
                else
                {
                    fprintf(stderr, "invalid outputpath extension type\n");
                    return -1;
                }

                if (req_format != PATHSTR("png") && req_format != PATHSTR("webp") && req_format != PATHSTR("jpg"))
                {
                    fprintf(stderr, "invalid format argument\n");
                    return -1;
                }

                std::vector<path_t> req_inputs(1, req_input);
                std::vector<path_t> req_outputs(1, req_output);

                LoadThreadParams ltp;
                ltp.scale = scale;
                ltp.jobs_load = jobs_load;
                ltp.input_files = req_inputs;
                ltp.output_files = req_outputs;

                g_done_lock.lock();
                g_done_count = 0;
                g_done_lock.unlock();

                ncnn::Thread load_thread(load, (void*)&ltp);
                load_thread.join();

                g_done_lock.lock();
                while (g_done_count < (int)req_inputs.size())
                {
                    g_done_condition.wait(g_done_lock);
                }
                g_done_lock.unlock();

                return 0;
            };

            if (opt.daemon_pipe.empty())
            {
                run_one_request(inputpath, outputpath, format);
            }
            else
            {
                for (;;)
                {
                    FILE* pipe_fp = fopen(opt.daemon_pipe.c_str(), "r");
                    if (!pipe_fp)
                    {
                        fprintf(stderr, "failed to open daemon pipe %s\n", opt.daemon_pipe.c_str());
                        break;
                    }

                    char* line = NULL;
                    size_t linecap = 0;
                    while (getline(&line, &linecap, pipe_fp) != -1)
                    {
                        std::istringstream iss(line);
                        std::vector<std::string> tokens;
                        std::string token;
                        while (iss >> token)
                            tokens.push_back(token);

                        if (tokens.empty())
                            continue;

                        std::vector<char*> cargs(tokens.size() + 1);
                        cargs[0] = (char*)"waifu2x-ncnn-vulkan";
                        for (size_t i = 0; i < tokens.size(); i++)
                            cargs[i + 1] = (char*)tokens[i].c_str();

                        Options req;
                        req.format = format;
                        if (parse_options((int)cargs.size(), cargs.data(), req, true) != 0)
                        {
                            fprintf(stderr, "warning: invalid daemon request, ignored\n");
                            continue;
                        }

                        run_one_request(req.inputpath, req.outputpath, req.format);
                    }

                    free(line);
                    fclose(pipe_fp);
                }
            }

            Task end;
            end.id = -233;

            for (int i=0; i<total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i=0; i<jobs_save; i++)
            {
                tosave.put(end);
            }

            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete waifu2x[i];
        }
        waifu2x.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
