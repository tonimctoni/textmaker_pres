#include "perceptron_timeseries_class.hpp"
#include "LSTM_class.hpp"
#include "softmax_timeseries_class.hpp"
#include <unordered_map>
#include <chrono>
#include <memory>
using namespace std;

void complete_with_wb(const string &start_string, const char *wb_filename)
{
    static constexpr size_t output_length=125;

    static constexpr size_t allowed_char_amount=46;
    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    static constexpr unsigned long first_mem_cell_size=512;
    static constexpr unsigned long second_mem_cell_size=256;
    static constexpr unsigned long third_mem_cell_size=128;
    static constexpr unsigned long output_mem_size=allowed_char_amount;

    //Initialize neural network
    using Block01=BaseTahnPerceptronBlock<input_size,reduced_input_size>;
    using Block02=BaseLSTMBlock<reduced_input_size, first_mem_cell_size>;
    using Block03=BaseLSTMBlock<first_mem_cell_size, second_mem_cell_size>;
    using Block04=BaseLSTMBlock<second_mem_cell_size, third_mem_cell_size>;
    using Block05=BaseSoftmaxBlock<third_mem_cell_size, output_mem_size>;
    unique_ptr<Block01> perceptronblock(new Block01(output_length));
    unique_ptr<Block02> lstmblock1(new Block02(output_length));
    unique_ptr<Block03> lstmblock2(new Block03(output_length));
    unique_ptr<Block04> lstmblock3(new Block04(output_length));
    unique_ptr<Block05> softmaxblock(new Block05(output_length));
    {
        ifstream in(wb_filename, std::ios::binary);
        assert(in.good());
        perceptronblock->only_wb_from_bin_file(in);
        lstmblock1->only_wb_from_bin_file(in);
        lstmblock2->only_wb_from_bin_file(in);
        lstmblock3->only_wb_from_bin_file(in);
        softmaxblock->only_wb_from_bin_file(in);
    }
    OneHot<input_size> X;
    auto calc_output=[&](size_t i) -> const Matrix<1,output_mem_size>&{
        perceptronblock->calc(X.get(), i);
        lstmblock1->calc(perceptronblock->get_output(i), i);
        lstmblock2->calc(lstmblock1->get_output(i), i);
        lstmblock3->calc(lstmblock2->get_output(i), i);
        softmaxblock->calc(lstmblock3->get_output(i), i);
        return softmaxblock->get_output(i);
    };

    //Setup the char_to_index and index_to_char mappings
    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);

    //Complete text
    {
        size_t i;
        for(i=0;i<start_string.size()-1;i++)
        {
            cout << start_string[i];
            X.set(char_to_index[start_string[i]]);
            calc_output(i);
        }

        cout << start_string[i];
        X.set(char_to_index[start_string[i]]);
        auto &Y=calc_output(i);
        size_t new_char_index=get_weighted_random_index(Y[0]);
        cout << index_to_char[new_char_index];
        X.set(new_char_index);
        for(i++;i<output_length;i++)
        {
            auto &Y=calc_output(i);
            new_char_index=get_weighted_random_index(Y[0]);
            cout << index_to_char[new_char_index];
            X.set(new_char_index);
        }
        cout << endl;
    }
}

int main()
{
    auto consists_of_valid_chars=[](const string &s){
        auto is_valid_char=[](char c){
            static const string valid_chars="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
            for(const char c_:valid_chars){
                if(c==c_)return true;
            }
            return false;
        };
        for(const char c_:s){
            if(not is_valid_char(c_)) return false;
        }
        return true;
    };

    string start_string;
    cout << "Enter the first part of the string: ";
    getline(cin, start_string);
    assert(consists_of_valid_chars(start_string));
    assert(start_string.size()<100);
    complete_with_wb(start_string, "tn.wab");
    complete_with_wb(start_string, "tr.wab");
    return 0;
}