#ifndef MEMORYCPU_H
#define MEMORYCPU_H

#include <unordered_map>
#include <vector>

namespace cpu
{
    // containers
    typedef struct
    {
        float *__restrict v;
        int size;
    } vtor;

#define CPU_VTOR_INIT  \
    {              \
        nullptr, 0 \
    }

    typedef struct
    {
        float *__restrict m;
        int rows;
        int cols;
    } matrix;

#define CPU_MATRIX_INIT   \
    {                 \
        nullptr, 0, 0 \
    }

    // fully connected layers scheduler
    typedef struct
    {
        vtor bias;
        matrix params;
        vtor layer_output;
    } f_layer_layout;

#define F_INIT                            \
    {                                     \
        CPU_VTOR_INIT, CPU_MATRIX_INIT, CPU_VTOR_INIT \
    }

    class fully_con_scheduler
    {
    public:
        class receiver
        {
        public:
            virtual void on_fully_receive(f_layer_layout layer) = 0;
        };

    private:
        int _size;    // fully connected layers total size
        float *_data; // fully connected layers data (params, bias)

        int _bias_max_size;                                     // max(biases sizes), need to multiply by 2
        float *_output_max;                                     // used in normal forward, double buffer shared by every layer (alternating buff)
        std::unordered_map<receiver *, f_layer_layout> _layers; // fully connected layers precise values
        std::vector<receiver *> _layers_order;                  // keep layers ordered

    private:
        bool _allocate_mem();
        void _free_mem();

    public:
        fully_con_scheduler();
        fully_con_scheduler(const fully_con_scheduler &rh) = delete;
        fully_con_scheduler(fully_con_scheduler &&rh);
        fully_con_scheduler &operator=(const fully_con_scheduler &rh) = delete;
        fully_con_scheduler &operator=(fully_con_scheduler &&rh);
        ~fully_con_scheduler();

        void request_layer(receiver *rec, int v_size, int m_rows, int m_cols);
        void assign_layers();
        float *get_ptr() const;
        int get_size() const;
    };

    // gradient layers scheduler
    typedef struct
    {
        vtor fx_bias;
        matrix fx_params;
        vtor layer_output;
        vtor fx_activations;
    } gradient_layer_layout;

#define CPU_G_INIT                                       \
    {                                                \
        CPU_VTOR_INIT, CPU_MATRIX_INIT, CPU_VTOR_INIT, CPU_VTOR_INIT \
    }

    class gradient_scheduler
    {
    public:
        class receiver
        {
        public:
            virtual void on_gradient_receive(gradient_layer_layout layer, float *fx_data, float *fx_accum) = 0;
        };

    private:
        int _size;
        float *_fx_data;  // gradient params, bias fxdata
        float *_fx_accum; // fxdata accumulator

        int _out_act_size;                                             // total layer outputs size + total fx activations size (2*sum(bias sizes))
        float *_output_activations;                                    // intermediate gradient forward results and activations derivatives, contiguous memory
        std::unordered_map<receiver *, gradient_layer_layout> _layers; // gradient layers precise values
        std::vector<receiver *> _layers_order;                         // keep layers ordered

    private:
        bool _allocate_mem();
        void _free_mem();

    public:
        gradient_scheduler();
        gradient_scheduler(const gradient_scheduler &rh) = delete;
        gradient_scheduler(gradient_scheduler &&rh);
        gradient_scheduler &operator=(const gradient_scheduler &rh) = delete;
        gradient_scheduler &operator=(gradient_scheduler &&rh);
        ~gradient_scheduler();

        void request_layer(receiver *rec, int v_size, int m_rows, int m_cols);
        void assign_layers();
        float *get_accum_ptr() const;
        float *get_fx_ptr() const;
        int get_size() const;
    };

    // batch scheduler
    typedef struct
    {
        float *input_batch;
        float *output_batch;
        int batch_size;
    } batch;

#define CPU_BATCH_INIT          \
    {                       \
        nullptr, nullptr, 0 \
    }

    typedef struct
    {
        int sample_input_size;
        int sample_output_size;
        int batch_num;
    } batch_data;

#define CPU_BATCH_DATA_INIT \
    {                   \
        0, 0, 0         \
    }

    class batch_scheduler
    {
    private:
        batch _batch;
        batch_data _batch_data;
        int _batch_size;
        int _data_num;
        std::vector<int> _samples_index; // random samples to draw each epoch
        int _it_until_epoch;         // it until new epoch
        int _batch_remainder;        // if data is not completely divisible by batch size
                                         // TODO include group norm
    private:
        void _free_mem();

    public:
        batch_scheduler();
        batch_scheduler(const batch_scheduler &rh) = delete;
        batch_scheduler(batch_scheduler &&rh);
        batch_scheduler &operator=(const batch_scheduler &rh) = delete;
        batch_scheduler &operator=(batch_scheduler &&rh);
        ~batch_scheduler();

        // no restrict used
        batch_data get_batch_data(int input_data_size, int output_data_size, int data_num, int batch_size);
        batch get_batch(const float *input_data, const float *output_data);
    };
}

#endif