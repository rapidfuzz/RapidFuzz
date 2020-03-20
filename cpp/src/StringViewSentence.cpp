#include <iterator> // std::iterator,std::input_iterator_tag
#include <iostream> // std::cout
#include <vector>
#include <string>
#include <string_view>

//template <typename T, std::size_t N>
class SampleVector
{
    std::vector<std::string_view> vector;
public:   
    class iterator : public std::iterator<std::input_iterator_tag, std::vector<std::string_view>::iterator>
    {
        std::vector<std::string_view>::iterator vector_ptr;
        std::string_view* char_ptr;
    public:
        iterator(std::vector<std::string_view>::iterator vector) : vector_ptr(vector) {

        }
        iterator& operator++() {
            ++vector_ptr;
            return *this;
        }
        iterator operator++(int) {
            iterator tmp(*this);
            operator++();
            return tmp;
        }
        bool operator==(const iterator& rhs) {
            return vector_ptr == rhs.vector_ptr;
        }
        bool operator!=(const iterator& rhs) { return vector_ptr != rhs.vector_ptr; }
        const char& operator*() {return *vector_ptr;} 
    };
    
    //SampleVector() {}

    SampleVector(std::vector<std::string_view> vec): vector(vec) {}
    
    iterator begin()
    {
        return iterator(vector.begin());
    }
    
    iterator end()
    {
        return iterator(vector.end());
    }
};

std::vector<std::string> test{"test", "test2"};
SampleVector vector("test");

int main()
{      
    for (int value: vector)
        std::cout << value << std::endl;
}