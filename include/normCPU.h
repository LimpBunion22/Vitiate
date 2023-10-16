#ifndef NORMCPU_H
#define NORMCPU_H

#include <defines.h>

namespace cpu::single
{
    class gradient_norm // for all gradient derivatives
    {
    private:
        float _lambda = 0.1f;
        int _sel_norm = net::DISABLE;
        bool _use_reg = false;

    private:
        float (gradient_norm::*_norm)(const float *__restrict, int) = nullptr;
        void (gradient_norm::*_update)(float, int, float, const float *__restrict, float *__restrict, int) = &gradient_norm::_sum;
        float _max_norm(const float *__restrict data, int size);
        float _abs_norm(const float *__restrict data, int size);
        float _modulo_norm(const float *__restrict data, int size);

        void _reg(float alpha, int batch_size, float norm, const float *__restrict fx_accum, float *__restrict data, int size); // regularization
        void _sum(float alpha, int batch_size, float norm, const float *__restrict fx_accum, float *__restrict data, int size);

    public:
        void select_norm(int norm);
        void use_reg(bool use);
        void set_lambda(float lambda);
        void norm(float alpha, const float *__restrict fx_accum, float *__restrict data, int size, int batch_size);

        gradient_norm() = default;
        gradient_norm(const gradient_norm &rh);
        gradient_norm(gradient_norm &&rh);
        gradient_norm &operator=(const gradient_norm &rh);
        gradient_norm &operator=(gradient_norm &&rh);
        ~gradient_norm() = default;
    };

    class instance_norm // per sample, no statistics
    {
    private:
        void (instance_norm::*_norm)(float *__restrict, int) = nullptr;
        float _max_clamp = 1.0f;
        float _min_clamp = 0.0f;

    private:
        void _min_max(float *__restrict data, int size);
        void _standarization(float *__restrict data, int size);

    public:
        void select_norm(int norm);
        void set_min_max_clamp(float min, float max);
        void norm(float *__restrict data, int size);
    };

    class group_norm // abstract class
    {
    public:
        virtual void select_norm(int norm) = 0;
        virtual void set_min_max_clamp(float min, float max) = 0;
        virtual void set_running_alpha(float alpha) = 0;
        virtual void norm(float *__restrict data, int total_size, int single_size) = 0;
        virtual void inference(float *__restrict data, int size) = 0;
    };

    class feat_norm : public group_norm // per feature (eg, input with 4 features, calculate min of feature 1 using all training data)
    {
    private:
        typedef struct
        {
            float a;
            float b;
        } norm_vals; // contiguous in memory

        void (feat_norm::*_norm)(float *__restrict, int, int) = nullptr;
        void (feat_norm::*_inference)(float *__restrict, int) = nullptr;
        std::vector<norm_vals> _running_mean_dev; // for use in inference (values per feature)
        std::vector<norm_vals> _running_min_max;  // for use in inference (values per feature)
        float _max_clamp = 1.0f;
        float _min_clamp = 0.0f;
        float _running_alpha = 0.99f;
        size_t _feats_num = 0;

    private:
        void _min_max(float *__restrict data, int total_size, int sample_size);
        void _min_max_inference(float *__restrict data, int size);
        void _standarization(float *__restrict data, int total_size, int sample_size);
        void _standarization_inference(float *__restrict data, int size);

    public:
        void select_norm(int norm) override;
        void set_min_max_clamp(float min, float max) override;
        void set_running_alpha(float alpha) override;
        void norm(float *__restrict data, int total_size, int sample_size) override; // sample size=number of feats
        void inference(float *__restrict data, int size) override;
    };

    class channel_norm : public group_norm // per channel (eg, input with three channels (RGB image), calculate mean of channel R using all training data)
    {
    private:
        typedef struct
        {
            float a;
            float b;
        } norm_vals; // contiguous in memory

        void (channel_norm::*_norm)(float *__restrict, int, int) = nullptr;
        void (channel_norm::*_inference)(float *__restrict, int) = nullptr;
        std::vector<norm_vals> _running_mean_dev; // for use in inference (values per feature)
        float _max_clamp = 1.0f;
        float _min_clamp = 0.0f;
        float _running_alpha = 0.99f;
        size_t _channels_num = 0;

    private:
        void _min_max(float *__restrict data, int total_size, int channel_size);
        void _min_max_inference(float *__restrict data, int size);
        void _standarization(float *__restrict data, int total_size, int channel_size);
        void _standarization_inference(float *__restrict data, int size);

    public:
        void select_norm(int norm) override;
        void set_min_max_clamp(float min, float max) override;
        void set_running_alpha(float alpha) override;
        void norm(float *__restrict data, int total_size, int channel_size) override;
        void inference(float *__restrict data, int size) override;
    };
}

#endif