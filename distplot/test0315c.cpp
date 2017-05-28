#include "perceptron_timeseries_class.hpp"
#include "LSTM_class.hpp"
#include "softmax_timeseries_class.hpp"
#include <unordered_map>
// #include <ctime>
#include <memory>
using namespace std;

template<unsigned long first_mem_cell_size, unsigned long second_mem_cell_size, unsigned long third_mem_cell_size>
class MyNet
{
private:
    static constexpr size_t allowed_char_amount=46;
    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    static constexpr unsigned long output_mem_size=allowed_char_amount;

    using Block01=BaseTahnPerceptronBlock<input_size,reduced_input_size>;
    using Block02=BaseLSTMBlock<reduced_input_size, first_mem_cell_size>;
    using Block03=BaseLSTMBlock<first_mem_cell_size, second_mem_cell_size>;
    using Block04=BaseLSTMBlock<second_mem_cell_size, third_mem_cell_size>;
    using Block05=BaseSoftmaxBlock<third_mem_cell_size, output_mem_size>;

    unique_ptr<Block01> perceptronblock;
    unique_ptr<Block02> lstmblock1;
    unique_ptr<Block03> lstmblock2;
    unique_ptr<Block04> lstmblock3;
    unique_ptr<Block05> softmaxblock;
public:
    MyNet(const char *filename)noexcept: perceptronblock(new Block01), lstmblock1(new Block02), lstmblock2(new Block03), lstmblock3(new Block04), softmaxblock(new Block05)
    {
        ifstream in(filename);
        assert(in.good());
        perceptronblock->only_wb_from_bin_file(in);
        lstmblock1->only_wb_from_bin_file(in);
        lstmblock2->only_wb_from_bin_file(in);
        lstmblock3->only_wb_from_bin_file(in);
        softmaxblock->only_wb_from_bin_file(in);
    }

    inline void set_time_steps(size_t time_steps) noexcept
    {
        perceptronblock->set_time_steps(time_steps);
        lstmblock1->set_time_steps(time_steps);
        lstmblock2->set_time_steps(time_steps);
        lstmblock3->set_time_steps(time_steps);
        softmaxblock->set_time_steps(time_steps);
    }

    inline const Matrix<1,output_mem_size>& calc(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        perceptronblock->calc(X, time_step);
        lstmblock1->calc(perceptronblock->get_output(time_step), time_step);
        lstmblock2->calc(lstmblock1->get_output(time_step), time_step);
        lstmblock3->calc(lstmblock2->get_output(time_step), time_step);
        softmaxblock->calc(lstmblock3->get_output(time_step), time_step);
        return softmaxblock->get_output(time_step);
    }
};

int main()
{
    MyNet<512,256,128> mynetn("outs/tn.wab");

    static constexpr size_t allowed_char_amount=46;
    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);

    static constexpr unsigned long input_size=allowed_char_amount;
    // static constexpr unsigned long output_mem_size=allowed_char_amount;


    static constexpr size_t output_size=125;
    string start_string;
    string end_stringn;

    print("Starting string:");
    getline(cin, start_string);
    OneHot<input_size> X;
    mynetn.set_time_steps(start_string.size()+output_size);

    ofstream out_file("test0315c.txt",std::ios_base::trunc);
    end_stringn.reserve(output_size);
    for(size_t i=0;i<start_string.size();i++)
    {
        if(i<start_string.size()-1)
        {
            X.set(char_to_index.at(start_string[i]));
            mynetn.calc(X.get(), i);
        }
        else
        {
            X.set(char_to_index.at(start_string[i]));
            auto& dist=mynetn.calc(X.get(), i)[0];
            for(auto &prob:dist) out_file << prob << endl;
            end_stringn.push_back(index_to_char.at(get_weighted_random_index(dist)));
        }
    }

    for(size_t i=start_string.size();i<start_string.size()+output_size-1;i++)
    {
        X.set(char_to_index.at(end_stringn.back()));
        auto& dist=mynetn.calc(X.get(), i)[0];
        for(auto &prob:dist) out_file << prob << endl;
        end_stringn.push_back(index_to_char.at(get_weighted_random_index(dist)));
    }

    print(start_string+end_stringn);
    out_file << end_stringn << endl;
    out_file.close();


    return 0;
}