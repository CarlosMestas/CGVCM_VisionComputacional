#ifndef RBG_H
#define RBG_H

#include <vector>

template<typename Type>

class RGB{

public:
    RGB();

    const Type& operator[](unsigned index) const{ return channels[index];}
    Type& operator[](unsigned index) { return channels[index];}

    using valueType = Type;

private:
    std::vector<Type> channels{};

};

template<typename Type>
RGB<Type>::RGB(): channels{std::vector<Type>(3, {})}{

}

#endif // RBG_H
