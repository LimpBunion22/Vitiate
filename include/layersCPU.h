#ifndef LAYERSCPU_H
#define LAYERSCPU_H

#include <memoryCPU.h>
#include <algosCPU.h>

namespace cpu
{
    // fully connected layer base class
    class fully_layer : public fully_con_scheduler::receiver,
                        public gradient_scheduler::receiver,
                        public single::dropout::receiver
    {
        friend class hidden_fully_layer;
        friend class out_fully_layer;
        friend class in_fully_layer;

    private:
        f_layer_layout _layer;
        gradient_layer_layout _gradient;
        single::activations _activations;
        vtor _dropout_mask; // all but output layer may use it. Default init
        // is used to determine if dropout must be used
        float *_fx_data;
        float *_fx_accum;

    protected:
        fully_layer *_next_layer;
        fully_layer *_prev_layer;

    private:
        // overrides
        void on_fully_receive(f_layer_layout layer) override;
        void on_gradient_receive(gradient_layer_layout layer, float *fx_data, float *fx_accum) override;
        void on_dropout_receive(vtor mask) override;
        void on_dropout_deactivate() override;

        fully_layer();
        fully_layer(const fully_layer &rh) = delete;
        fully_layer(fully_layer &&rh);
        fully_layer &operator=(const fully_layer &rh) = delete;
        fully_layer &operator=(fully_layer &&rh);

    public:
        virtual ~fully_layer() = default;

        // layer data mangement
        void get_next_layer(fully_layer *layer);
        void get_prev_layer(fully_layer *layer);
        void random_set_fully_layer();

        // activations
        void set_activation(int activation, int soft_size);

        // forward
        virtual void run_layer_forward() = 0;

        // gradient
        virtual void run_layer_gradient_forward() = 0;
        virtual void run_layer_gradient(int batch_size) = 0;
    };

    // hidden fully connected layer
    class hidden_fully_layer : public fully_layer
    {
    public:
        hidden_fully_layer() = default;
        hidden_fully_layer(const hidden_fully_layer &rh) = delete;
        hidden_fully_layer(hidden_fully_layer &&rh);
        hidden_fully_layer &operator=(const hidden_fully_layer &rh) = delete;
        hidden_fully_layer &operator=(hidden_fully_layer &&rh);
        ~hidden_fully_layer() = default;

        void run_layer_forward() override;
        void run_layer_gradient_forward() override;
        void run_layer_gradient(int batch_size) override;
    };

    // output fully connected layer
    class out_fully_layer : public fully_layer
    {
    private:
        vtor _R;
        vtor _output;
        float *_fwd_output; // output host buffer to copy normal fwd result to

    public:
        out_fully_layer();
        out_fully_layer(const out_fully_layer &rh) = delete;
        out_fully_layer(out_fully_layer &&rh);
        out_fully_layer &operator=(const out_fully_layer &rh) = delete;
        out_fully_layer &operator=(out_fully_layer &&rh);
        ~out_fully_layer() = default;

        void run_layer_forward() override;
        void run_layer_gradient_forward() override;
        void run_layer_gradient(int batch_size) override;

        void set_R(vtor R);
        void set_sample_output(vtor output);
        void set_output_buff(float *output);
        int get_output_size() const;
    };

    // input fully connected layer
    class in_fully_layer : public fully_layer
    {
    private:
        const float *_fwd_input; // used to point to data to be copied from in normal forward
        vtor _input_buffer;      // used for deep copy in normal forward
        vtor _input_ptr;         // used for accessing

    private:
        void _free_mem();

    public:
        in_fully_layer() = delete;
        in_fully_layer(int input_size);
        in_fully_layer(const in_fully_layer &rh) = delete;
        in_fully_layer(in_fully_layer &&rh);
        in_fully_layer &operator=(const in_fully_layer &rh) = delete;
        in_fully_layer &operator=(in_fully_layer &&rh);
        ~in_fully_layer();

        void run_layer_forward() override;
        void run_layer_gradient_forward() override;
        void run_layer_gradient(int batch_size) override;

        void set_input(vtor input);         // used in gradient fwd
        void set_input(const float *input); // used in normal fwd
    };
}

#endif