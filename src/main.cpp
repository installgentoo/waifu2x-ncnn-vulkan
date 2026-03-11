// waifu2x implemented with ncnn library

#include <string.h>
#include <future>
#include <queue>
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
    fprintf(stdout, "  -D pipe-file         daemon mode, read per-job args from named pipe\n");
}

class Args
{
public:
    path_t inputpath;
    path_t outputpath;
    int noise;
    int scale;
    std::vector<int> tilesize;
    path_t model;
    std::vector<int> gpuid;
    int jobs_load;
    std::vector<int> jobs_proc;
    int jobs_save;
    int verbose;
    int tta_mode;
    path_t format;
    path_t daemon_pipe;
    path_t daemon_status_path;

    Args()
        : noise(0), scale(2), model(PATHSTR("models-cunet")), jobs_load(1), jobs_save(2), verbose(0), tta_mode(0), format(PATHSTR("png"))
    {
    }
};

class ParseMeta
{
public:
    bool daemon_canonical;
    bool daemon_request_valid;

    ParseMeta()
        : daemon_canonical(true), daemon_request_valid(true)
    {
    }
};

static int parse_args(int argc, char** argv, Args& args, ParseMeta& meta)
{
    int opt;
    bool has_input = false;
    bool has_output = false;

    optind = 1;
    opterr = 0;
    while ((opt = getopt(argc, argv, "i:o:n:s:t:m:g:j:f:D:p:vxh")) != -1)
    {
        switch (opt)
        {
        case 'i':
            args.inputpath = optarg;
            has_input = true;
            break;
        case 'o':
            args.outputpath = optarg;
            has_output = true;
            break;
        case 'n':
            args.noise = atoi(optarg);
            break;
        case 's':
            args.scale = atoi(optarg);
            break;
        case 't':
            args.tilesize = parse_optarg_int_array(optarg);
            break;
        case 'm':
            args.model = optarg;
            meta.daemon_canonical = false;
            break;
        case 'g':
            args.gpuid = parse_optarg_int_array(optarg);
            meta.daemon_canonical = false;
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &args.jobs_load, &args.jobs_save);
            args.jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            meta.daemon_canonical = false;
            break;
        case 'f':
            args.format = optarg;
            break;
        case 'D':
            args.daemon_pipe = optarg;
            meta.daemon_canonical = false;
            break;
        case 'p':
            args.daemon_status_path = optarg;
            break;
        case 'v':
            args.verbose = 1;
            break;
        case 'x':
            args.tta_mode = 1;
            meta.daemon_canonical = false;
            break;
        default:
            return -1;
        }
    }

    if (!has_input || !has_output)
    {
        meta.daemon_request_valid = false;
    }

    if (args.noise < -1 || args.noise > 3)
    {
        meta.daemon_request_valid = false;
    }

    if (!(args.scale == 1 || args.scale == 2 || args.scale == 4 || args.scale == 8 || args.scale == 16 || args.scale == 32))
    {
        meta.daemon_request_valid = false;
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

    path_t daemon_status_path;  // non-empty only in daemon mode
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

class LoadThreadParams
{
public:
    int scale;
    int jobs_load;
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

        FILE* fp = fopen(imagepath.c_str(), "rb");
        if (fp)
        {
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
                    pixeldata = jpeg_load(filedata, length, &w, &h, &c);
                    if (!pixeldata)
                    {
                        pixeldata = png_load(filedata, length, &w, &h, &c);
                    }
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
                fprintf(stderr, "image %s has alpha channel ! %s will output %s\n", imagepath.c_str(), imagepath.c_str(), output_filename2.c_str());
            }

            toproc.put(v);
        }
        else
        {
            fprintf(stderr, "decode image %s failed\n", imagepath.c_str());
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
            success = png_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, (const unsigned char*)v.outimage.data);
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
            success = jpeg_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, (const unsigned char*)v.outimage.data);
        }
        if (success)
        {
            if (verbose)
            {
                fprintf(stdout, "%s -> %s done\n", v.inpath.c_str(), v.outpath.c_str());
            }
        }
        else
        {
            fprintf(stderr, "encode image %s failed\n", v.outpath.c_str());
        }

        if (!v.daemon_status_path.empty())
        {
            FILE* statusfp = fopen(v.daemon_status_path.c_str(), "wb");
            if (statusfp)
            {
                fprintf(statusfp, "%s\n", success ? "OK" : "FAIL");
                fclose(statusfp);
            }
            else
            {
                fprintf(stderr, "warning: failed to write daemon status %s\n", v.daemon_status_path.c_str());
            }
        }
    }

    return 0;
}


int main(int argc, char** argv)
{
    Args args;
    ParseMeta meta;
    if (parse_args(argc, argv, args, meta) != 0)
    {
        print_usage();
        return -1;
    }

    const bool daemon_mode = !args.daemon_pipe.empty();

    if ((args.inputpath.empty() || args.outputpath.empty()) && !daemon_mode)
    {
        print_usage();
        return -1;
    }

    int noise = args.noise;
    int scale = args.scale;
    std::vector<int> tilesize = args.tilesize;
    path_t model = args.model;
    std::vector<int> gpuid = args.gpuid;
    int jobs_load = args.jobs_load;
    std::vector<int> jobs_proc = args.jobs_proc;
    int jobs_save = args.jobs_save;
    int verbose = args.verbose;
    int tta_mode = args.tta_mode;
    path_t format = args.format;

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

    if (!daemon_mode && !path_is_directory(args.outputpath))
    {
        // guess format from outputpath no matter what format argument specified
        path_t ext = get_file_extension(args.outputpath);

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
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg"))
    {
        fprintf(stderr, "invalid format argument\n");
        return -1;
    }

    if (!daemon_mode && (path_is_directory(args.inputpath) || path_is_directory(args.outputpath)))
    {
        fprintf(stderr, "inputpath and outputpath must be file paths\n");
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
    int jobs_proc_per_gpu[16] = {0};
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
            jobs_proc_per_gpu[gpuid[i]] += jobs_proc[i];
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

        // multiple gpu jobs share the same heap
        heap_budget /= jobs_proc_per_gpu[gpuid[i]];

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

            if (daemon_mode)
            {
                FILE* pipefp = fopen(args.daemon_pipe.c_str(), "rb");
                if (!pipefp)
                {
                    fprintf(stderr, "open daemon pipe %s failed\n", args.daemon_pipe.c_str());
                }
                else
                {
                    auto write_fail_status = [](const path_t& status_path)
                    {
                        if (status_path.empty())
                            return;
                        FILE* statusfp = fopen(status_path.c_str(), "wb");
                        if (statusfp)
                        {
                            fprintf(statusfp, "FAIL\n");
                            fclose(statusfp);
                        }
                    };

                    auto process_daemon_request = [&](const std::vector<std::string>& request_tokens)
                    {
                        char* line_argv[256] = {0};
                        int line_argc = 1;
                        line_argv[0] = argv[0];

                        for (int i=0; i<(int)request_tokens.size() && line_argc < 255; i++)
                        {
                            line_argv[line_argc++] = (char*)request_tokens[i].c_str();
                        }

                        if (line_argc <= 1)
                            return;

                        Args task_args = args;
                        task_args.inputpath.clear();
                        task_args.outputpath.clear();
                        ParseMeta task_meta;
                        if (parse_args(line_argc, line_argv, task_args, task_meta) != 0)
                        {
                            fprintf(stderr, "invalid daemon arguments line ignored\n");
                            write_fail_status(task_args.daemon_status_path);
                            return;
                        }

                        task_args.model = model;
                        task_args.gpuid = gpuid;
                        task_args.jobs_load = jobs_load;
                        task_args.jobs_proc = jobs_proc;
                        task_args.jobs_save = jobs_save;
                        task_args.tta_mode = tta_mode;

                        bool fixed_model_conflict = task_args.noise != noise || (task_args.scale == 1) != (scale == 1);
                        bool daemon_request_valid = task_meta.daemon_request_valid && (task_args.tilesize.empty() || task_args.tilesize.size() == (size_t)use_gpu_count);

                        if (!daemon_request_valid)
                        {
                            fprintf(stderr, "warning: invalid daemon request; use valid -i/-o/-n/-s/-t values in each pipe line\n");
                            write_fail_status(task_args.daemon_status_path);
                            return;
                        }

                        if (fixed_model_conflict)
                        {
                            task_args.noise = noise;
                            task_args.scale = scale;
                        }

                        if (!task_meta.daemon_canonical || fixed_model_conflict)
                        {
                            fprintf(stderr, "warning: arguments -m/-g/-j/-x/-D and model-changing -n/-s must be supplied when starting daemon mode\n");
                        }

                        for (int i=0; i<use_gpu_count; i++)
                        {
                            waifu2x[i]->tilesize = task_args.tilesize.empty() ? tilesize[i] : task_args.tilesize[i];
                        }

                        // load image inline
                        unsigned char* pixeldata = 0;
                        int w = 0, h = 0, c = 0;
                        {
                            FILE* fp = fopen(task_args.inputpath.c_str(), "rb");
                            if (fp)
                            {
                                fseek(fp, 0, SEEK_END);
                                int length = ftell(fp);
                                rewind(fp);
                                unsigned char* filedata = (unsigned char*)malloc(length);
                                if (filedata)
                                {
                                    fread(filedata, 1, length, fp);
                                    pixeldata = webp_load(filedata, length, &w, &h, &c);
                                    if (!pixeldata)
                                        pixeldata = jpeg_load(filedata, length, &w, &h, &c);
                                    if (!pixeldata)
                                        pixeldata = png_load(filedata, length, &w, &h, &c);
                                    free(filedata);
                                }
                                fclose(fp);
                            }
                        }

                        if (!pixeldata)
                        {
                            fprintf(stderr, "decode image %s failed\n", task_args.inputpath.c_str());
                            write_fail_status(task_args.daemon_status_path);
                            return;
                        }

                        Task v;
                        v.id = 0;
                        v.scale = task_args.scale;
                        v.inpath = task_args.inputpath;
                        v.outpath = task_args.outputpath;
                        v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
                        v.daemon_status_path = task_args.daemon_status_path;

                        path_t ext = get_file_extension(v.outpath);
                        if (c == 4 && (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")))
                        {
                            v.outpath = v.outpath + PATHSTR(".png");
                            fprintf(stderr, "image %s has alpha channel ! will output %s\n", task_args.inputpath.c_str(), v.outpath.c_str());
                        }

                        toproc.put(v);
                    };

                    std::vector<std::string> request_tokens;
                    std::string current_field;

                    for (;;)
                    {
                        int ch = fgetc(pipefp);
                        if (ch == EOF)
                            break;

                        if (ch != '\0')
                        {
                            current_field.push_back((char)ch);
                            continue;
                        }

                        if (!current_field.empty())
                        {
                            request_tokens.push_back(current_field);
                            current_field.clear();
                            continue;
                        }

                        if (!request_tokens.empty())
                        {
                            process_daemon_request(request_tokens);
                            request_tokens.clear();
                        }
                    }

                    if (!current_field.empty())
                    {
                        request_tokens.push_back(current_field);
                    }
                    if (!request_tokens.empty())
                    {
                        process_daemon_request(request_tokens);
                    }

                    fclose(pipefp);
                }
            }
            else
            {
                // non-daemon: single file, original simple path
                LoadThreadParams ltp;
                ltp.scale = scale;
                ltp.jobs_load = jobs_load;
                ltp.input_files.push_back(args.inputpath);
                ltp.output_files.push_back(args.outputpath);

                ncnn::Thread load_thread(load, (void*)&ltp);
                load_thread.join();
            }

            Task end;
            end.id = -233;

            for (int i=0; i<total_jobs_proc; i++)
                toproc.put(end);

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
