#pragma once

#ifndef VT_MACHINE_H
#define VT_MACHINE_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <variant>
#include <optional>
#include <iostream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <type_traits>

/*
 * some common interfaces used by all compoents
 */
#define vt_assert(Expr, Msg) \
    vt::_M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define vt_panic(Msg) \
    vt::_M_Panic(__FILE__, __LINE__, Msg)

#define vt_fatal_error() \
    vt::_M_Panic(__FILE__, __LINE__, "Fatal error, can't be here")

namespace vt {

const unsigned int VERSION = 0x000101;    // v0.1.1

// some common help functions
inline void _M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "**Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

inline void _M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "**Panic:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
}

template <typename TensorT, typename ContextT> struct Enviroment;

// TensorT 的接口基类：栈上张量类型须继承它并实现 to_string()（Echo/Dump 打印
// 张量 cell 用）。选继承而非"探测成员是否存在"：重写即签名检查，返回类型也被
// 钉死成 std::string；约束在 Enviroment 入口 static_assert（见下）把关。
struct TensorType {
    virtual std::string to_string() = 0;
    virtual ~TensorType() {}     // 多态基类惯例：虚析构（vt 内不经基类指针 delete）
};

// target number type
template <typename TensorT>
struct Cell {
    using tensor_t = std::shared_ptr<TensorT>;

    enum CellType {
        T_Number,
        T_String,
        T_Tensor,
    };
    const CellType type_;
    std::variant<double, const std::string, tensor_t> v_;

    // constructors
    Cell() : type_(T_Number), v_(0.0f) {}
    Cell(double value): type_(T_Number), v_(value) {}
    Cell(const std::string& value): type_(T_String), v_(value) {}
    Cell(tensor_t value) : type_(T_Tensor), v_(value) {}

    // fast access
    const std::string& as_string() {
        vt_assert(type_ == T_String, "Cell type can't convert to string!");
        return std::get<1>(v_);
    }
    bool as_boolean() {
        vt_assert(type_ == T_Number, "Cell type can't convert to boolean!");
        auto num = std::get<0>(v_);
        if ( num == 0.0) {
            return false;
        }
        return true;
    }
    double as_number() {
        vt_assert(type_ == T_Number, "Cell type can't convert to number!");
        return std::get<0>(v_);
    }
    tensor_t as_tensor() {
        vt_assert(type_ == T_Tensor, "Cell type can't convert to vector!");
        return std::get<2>(v_);
    }
    bool is_number() {
        if ( type_ == T_Number ) {
            return true;
        }
        return false;
    }
    bool is_string() {
        if ( type_ == T_String ) {
            return true;
        }
        return false;
    }
    bool is_tensor() {
        if ( type_ == T_Tensor ) {
            return true;
        }
        return false;
    }
};

template <typename TensorT>
std::ostream& operator<<(std::ostream& os, Cell<TensorT>& c) {
    if ( c.type_ == Cell<TensorT>::T_String ) {
        os << "S:" << c.as_string();
    } else if ( c.type_ == Cell<TensorT>::T_Number ) {
        os << "N:" << std::fixed << c.as_number();
    } else {
        auto t = c.as_tensor();
        if ( t != nullptr ) {
            os << "T: (" << t->to_string() << " )";
        } else {
            os << "T: (null)";
        }
    }
    return os;
}

// Stack & Hash
template <typename TensorT>
struct Stack {
    Stack() {}
    ~Stack() {}

    size_t size() {
        return data_.size();
    }
    void clear() {
        data_.clear();
    }

    Cell<TensorT>& top() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        return data_.back();
    }
    Cell<TensorT> pop() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret;
    }
    void drop() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        data_.pop_back();
    }
    void dup() {
        data_.push_back( top() );
    }
    void dup2() {
        auto a = pop();
        auto b = pop();
        data_.push_back(b);
        data_.push_back(a);
        data_.push_back(b);
        data_.push_back(a);
    }
    void swap() {
        auto a = pop();
        auto b = pop();
        data_.push_back(a);
        data_.push_back(b);
    }
    void rot() {
        auto a = pop();
        auto b = pop();
        auto c = pop();
        data_.push_back(b);
        data_.push_back(a);
        data_.push_back(c);
    }
    void rev(int n) {
        if ( n == -1) {
            data_.reverse();
            return;
        }
        if ( n < 0 ) {
           vt_panic("reverse can't support negtive number!");
        }

        std::vector<Cell<TensorT>> temp;
        for(int i = 0; i < n; i++) {
            temp.push_back( pop() );
        }
        for(int i = 0; i < n; i++) {
            data_.push_back( temp[i] );
        }
    }
    double pop_number() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_number();
    }
    std::vector<double> pop_number_list() {
        size_t s = (size_t) pop_number();
        std::vector<double> ret(s, 0.0);
        for (size_t i = 0; i < s ; i++) {
            ret[s - 1 - i] = pop_number();
        }
        return ret;
    }
    const std::string pop_string() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_string();
    }
    std::shared_ptr<TensorT> pop_tensor() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_tensor();
    }
    bool pop_boolean() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_boolean();
    }
    void push(Cell<TensorT> cell) {
        data_.push_back(cell);
    }
    void push_number(double n) {
        data_.push_back( Cell<TensorT>(n) );
    }
    void push_number_list(std::vector<double>& list) {
        for (size_t i = 0; i < list.size(); i++) {
            push_number( list[i] );
        }
        push_number( list.size() );
    }
    void push_tensor(std::shared_ptr<TensorT> t) {
        data_.push_back( Cell<TensorT>(t) );
    }
    void push_string(const std::string& str) {
        data_.push_back( Cell<TensorT>(str) );
    }

private:
    std::list< Cell<TensorT> > data_;

    template <typename T, typename C>
    friend struct BuiltinOperator;
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, Stack<U>& stack);
};

template <typename TensorT>
std::ostream& operator<<(std::ostream& os, Stack<TensorT>& stack) {
    int i = 0;
    os << "----STACK(" << stack.size() << ")----" << std::endl;
    for (auto it = stack.data_.begin(); it != stack.data_.end(); it++, i++) {
        os << "==>" << i << " " << *it << std::endl;
    }
    os << "----" << std::endl;
    return os;
}

template <typename TensorT>
struct Hash {
    using tensor_t = std::shared_ptr<TensorT>;
    using Item = std::variant<double, std::string, tensor_t>;
    Hash() {
    }
    ~Hash() {}

    Item find(const std::string& name) {
        if ( map_.find(name) == map_.end() ) {
            std::cerr << "Find: " << name << std::endl;
            vt_panic("Can't find value for name!");
        }
        return map_[name];
    }

    double find_number(const std::string& name) {
        auto item = find(name);
        return std::get<0>(item);
    }

    std::string find_string(const std::string& name) {
        auto item = find(name);
        return std::get<1>(item);
    }

    tensor_t find_tensor(const std::string& name) {
        auto item = find(name);
        return std::get<2>(item);
    }

    void set(const std::string& name, Item item) {
        map_[name] = item;
    }

    void drop(const std::string& name) {
        map_.erase(name);
    }

    static Cell<TensorT> Item2Cell( Item item ) {
        if ( item.index() == 0 ) {
            return Cell<TensorT>( std::get<0>(item) );
        } else if ( item.index() == 1 ) {
            return Cell<TensorT>( std::get<1>(item) );
        }
        return Cell<TensorT>( std::get<2>(item) );
    }

    static Item Cell2Item( Cell<TensorT>& cell ) {
        Item ret;
        if ( cell.type_ == Cell<TensorT>::T_Number ) {
            ret = cell.as_number();
        } else if ( cell.type_ == Cell<TensorT>::T_String ) {
            ret = cell.as_string();
        } else {
            ret = cell.as_tensor();
        }
        return ret;
    }

private:
    std::map<const std::string, Item> map_;
};

struct WordCode {
    enum {
        Number,
        String,
        Builtin,
        Native,
        User,
    } type_;

    std::string str_;
    double num_;

    static WordCode new_number(double n) {
        WordCode wc;
        wc.type_ = WordCode::Number;
        wc.num_ = n;
        return wc;
    }
    static WordCode new_string(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::String;
        wc.str_ = v;
        return wc;
    }
    static WordCode new_builtin(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::Builtin;
        wc.str_ = v;
        return wc;
    }
    static WordCode new_native(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::Native;
        wc.str_ = v;
        return wc;
    }
    static WordCode new_user(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::User;
        wc.str_ = v;
        return wc;
    }
};

struct WordByte {
    enum _WordByteType_ {
        Number,
        String,
        BuiltinOperator,
        Native,
    } type_;

    std::string str_;
    size_t idx_;
    double num_;

    WordByte(double num) : type_(Number) {
        num_ = num;
    }
    WordByte(const std::string& str) : type_(String) {
        str_ = str;
    }
    WordByte(_WordByteType_ t, size_t i): type_(t) {
        idx_ = i;
    }
    WordByte(_WordByteType_ t): type_(t) {
    }

};

inline std::ostream& operator<<(std::ostream& os, const WordCode& c) {
    if ( c.type_ == WordCode::String ) {
        os << "S:" << c.str_;
    } else if ( c.type_ == WordCode::Number ) {
        os << "N:" << c.num_;
    } else if ( c.type_ == WordCode::Builtin ) {
        os << "B:" << c.str_;
    } else if ( c.type_ == WordCode::Native ) {
        os << "NA:" << c.str_;
    } else {
        os << "U:" << c.str_;
    }
    return os;
}

// two type local function
template <typename TensorT, typename ContextT>
struct BuiltinOperator {
    virtual ~BuiltinOperator() {
    }
    virtual int run(Enviroment<TensorT, ContextT>* env) = 0;
};
template <typename TensorT>
struct NativeWord {
    virtual ~NativeWord() {
    }
    virtual void run(Stack<TensorT>& stack) = 0;
};

template <typename TensorT, typename ContextT>
using NativeCreator = NativeWord<TensorT>* (Enviroment<TensorT, ContextT>&);
using UserWord = std::vector<WordCode>;
using UserBinary = std::vector<WordByte>;

// native word 注册宏：所在模板类的模板参数须命名为 TensorT / ContextT
//（宏体内直接引用这两个名字）。LR = 不依赖 Enviroment，CTX = 构造时捕获 ctx()。
#define NWORD_CREATOR_DEFINE_LR(CLS)                                            \
    static vt::NativeWord<TensorT>* creator(vt::Enviroment<TensorT, ContextT>& env) { \
        vt::NativeWord<TensorT>* wd = new CLS();                                \
        return wd;                                                              \
    }

#define NWORD_CREATOR_DEFINE_CTX(CLS)                                           \
    ContextT* ctx_;                                                             \
    CLS(vt::Enviroment<TensorT, ContextT>& env) : ctx_(env.ctx()) {             \
    }                                                                           \
    static vt::NativeWord<TensorT>* creator(vt::Enviroment<TensorT, ContextT>& env) { \
        vt::NativeWord<TensorT>* wd = new CLS(env);                             \
        return wd;                                                              \
    }

template <typename TensorT, typename ContextT>
struct DaG {
    DaG() = delete;
    DaG(Enviroment<TensorT, ContextT>* env) : env_(env) {}

    ~DaG() {
        for (size_t i = 0; i < natives_.size(); i++) {
            delete natives_[i];
        }
        for (size_t i = 0; i < builtins_.size(); i++) {
            delete builtins_[i];
        }
    }

private:
    // borned from
    const Enviroment<TensorT, ContextT>* env_;

    // linked and resources
    UserBinary binary_;
    std::vector<NativeWord<TensorT>*> natives_;
    std::vector<BuiltinOperator<TensorT, ContextT>*> builtins_;

    friend struct Enviroment<TensorT, ContextT>;
};

// Enviroment 为模板类，两个类型均由使用方提供（vt 不定义默认类型）：
//   TensorT  —— 栈上张量类型，须继承上面的 TensorType 并实现 to_string()
//   ContextT —— 计算上下文，vt 只存指针透传，无接口要求
template <typename TensorT, typename ContextT>
struct Enviroment {
    // 接口把关：TensorT 不继承 TensorType 时，直接报在使用方声明那一行
    static_assert(std::is_base_of_v<TensorType, TensorT>,
                  "TensorT must inherit vt::TensorType (to_string() required by Echo/Dump)");

    Enviroment(ContextT* ctx);
    ~Enviroment() {
    }

    void insert_native_word(const std::string& name, NativeCreator<TensorT, ContextT>* fn) {
        if ( native_words_.find(name) != native_words_.end() ) {
            vt_panic("Can't insert native word with same name!");
        }
        native_words_[name] = fn;
    }

    DaG<TensorT, ContextT>* build(const std::string& txt) {
        DaG<TensorT, ContextT>* dag = new DaG<TensorT, ContextT>(this);

        UserWord myCode = compile(txt);
        linking(*dag, myCode);
        return dag;
    }
    void run(DaG<TensorT, ContextT>* dag) {
        vt_assert( dag->env_ == this, "Can't be here!");
        run_(dag);
    }

    void execute(const std::string& txt) {
        DaG<TensorT, ContextT> dag(this);

        UserWord myCode = compile(txt);
        linking(dag, myCode);
        run_(&dag);
    }

    Stack<TensorT>& stack() {
        return stack_;
    }
    Hash<TensorT>& hash() {
        return hash_;
    }

    ContextT* ctx() {
        return ctx_;
    }

private:
    void run_(DaG<TensorT, ContextT>* dag) {
        auto& binary_ = dag->binary_;
        auto& builtins_ = dag->builtins_;
        auto& natives_ = dag->natives_;

        for ( size_t pc = 0; pc < binary_.size(); ) {
            auto& byte = binary_[pc];
            int steps = 1;
            switch( byte.type_ ) {
                case WordByte::Number:
                    stack_.push_number( byte.num_ );
                    break;

                case WordByte::String:
                    stack_.push_string( byte.str_ );
                    break;

                case WordByte::BuiltinOperator:
                    steps = builtins_[ byte.idx_ ]->run( this );
                    break;

                case WordByte::Native:
                    natives_[ byte.idx_ ]->run( stack_ );
                    break;

                default:
                    vt_panic("Runing binary error can't bere!");
                    break;
            }
            vt_assert( steps > 0, " DAG don't support loop ");
            pc += steps;
        }
    }

    NativeWord<TensorT>* create_native(const std::string& name) {
        if ( native_words_.find(name) != native_words_.end() ) {
            return native_words_[name](*this);
        }
        vt_panic("Call a un registered native word!");
        return nullptr;
    }

    UserWord& get_user(const std::string& name) {
        if ( user_words_.find(name) == user_words_.end() ) {
            vt_panic("Call a un registered native word!");
        }
        return user_words_[name];
    }
    UserWord compile(const std::string& txt);
    void linking(DaG<TensorT, ContextT>& dag, UserWord& word);

    void load_base_words();
private:
    // compiled
    std::map<std::string, UserWord> user_words_;
    std::map<std::string, NativeCreator<TensorT, ContextT>*> native_words_;

    // runtime
    Stack<TensorT> stack_;
    Hash<TensorT> hash_;

    // global
    ContextT* ctx_;
};

template <typename TensorT, typename ContextT>
Enviroment<TensorT, ContextT>::Enviroment(ContextT* ctx) : ctx_(ctx) {
    load_base_words();
}

namespace builtin {
    template <typename TensorT, typename ContextT>
    struct BuiltinGet : public BuiltinOperator<TensorT, ContextT> {
        int run(Enviroment<TensorT, ContextT>* env) override {
            auto& hash = env->hash();
            auto& stack = env->stack();

            auto name = stack.pop_string();
            auto value = hash.find(name);
            stack.push( Hash<TensorT>::Item2Cell(value) );
            return 1;
        }
    };

    template <typename TensorT, typename ContextT>
    struct BuiltinSet : public BuiltinOperator<TensorT, ContextT> {
        int run(Enviroment<TensorT, ContextT>* env) override {
            auto& hash = env->hash();
            auto& stack = env->stack();

            auto name = stack.pop_string();
            Cell<TensorT> cell = stack.pop();
            hash.set(name, Hash<TensorT>::Cell2Item(cell));
            return 1;
        }
    };

    template <typename TensorT, typename ContextT>
    struct BuiltinRemove : public BuiltinOperator<TensorT, ContextT> {
        BuiltinRemove() {
        }
        int run(Enviroment<TensorT, ContextT>* env) override {
            auto& hash = env->hash();
            auto& stack = env->stack();

            auto name = stack.pop_string();
            hash.drop(name);
            return 1;
        }
    };

    template <typename TensorT, typename ContextT>
    struct BuiltinJNZ : public BuiltinOperator<TensorT, ContextT> {
        BuiltinJNZ() {
        }
        int run(Enviroment<TensorT, ContextT>* env) override {
            auto& stack = env->stack();

            int steps = stack.pop_number();
            int cond = stack.pop_number();
            if ( cond ) {
                return steps;
            }
            return 1;
        }
    };

    template <typename TensorT, typename ContextT>
    struct BuiltinJZ : public BuiltinOperator<TensorT, ContextT> {
        BuiltinJZ() {
        }
        int run(Enviroment<TensorT, ContextT>* env) override {
            auto& stack = env->stack();

            int steps = stack.pop_number();
            int cond = stack.pop_number();
            if ( !cond ) {
                return steps;
            }
            return 1;
        }
    };
}

template <typename TensorT, typename ContextT>
void Enviroment<TensorT, ContextT>::linking(DaG<TensorT, ContextT>& dag, UserWord& word) {
    auto& builtins_ = dag.builtins_;
    auto& natives_ = dag.natives_;
    auto& binary_ = dag.binary_;

    for(size_t i = 0; i < word.size(); i++) {
        auto code = word[i];
        switch( code.type_ ) {
            case WordCode::Number :
                binary_.push_back( WordByte( code.num_ ) );
                break;

            case WordCode::String :
                binary_.push_back( WordByte( code.str_  ) );
                break;

            case WordCode::Builtin :
                {
                    BuiltinOperator<TensorT, ContextT>* op = nullptr;
                    if ( code.str_ == "@" ) {
                        op = new builtin::BuiltinGet<TensorT, ContextT>();
                    } else if ( code.str_ == "!" ) {
                        op = new builtin::BuiltinSet<TensorT, ContextT>();
                    } else if ( code.str_ == "!!" ) {
                        op = new builtin::BuiltinRemove<TensorT, ContextT>();
                    } else if ( code.str_ == "jnz" ) {
                        op = new builtin::BuiltinJNZ<TensorT, ContextT>();
                    } else if ( code.str_ == "jz" ) {
                        op = new builtin::BuiltinJZ<TensorT, ContextT>();
                    } else {
                        vt_panic("Find an unsupoorted builtin operator!");
                    }
                    size_t idx = builtins_.size();
                    builtins_.push_back(op);
                    binary_.push_back( WordByte( WordByte::BuiltinOperator, idx) );
                }
                break;

            case WordCode::Native :
                binary_.push_back( WordByte(WordByte::Native, natives_.size() ));
                natives_.push_back( create_native(code.str_));
                break;

            case WordCode::User :
                UserWord& new_word = get_user( code.str_ );
                linking(dag, new_word);
                break;
        }
    }
}

namespace base {
    template <typename TensorT, typename ContextT>
    struct Exit : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            exit(0);
        }
        NWORD_CREATOR_DEFINE_LR(Exit)
    };

    template <typename TensorT, typename ContextT>
    struct Dump : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            std::cerr << stack << std::endl;
        }
        NWORD_CREATOR_DEFINE_LR(Dump)
    };

    template <typename TensorT, typename ContextT>
    struct Echo : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            Cell<TensorT> cell = stack.pop();
            std::cerr << cell << std::endl;
        }
        NWORD_CREATOR_DEFINE_LR(Echo)
    };

    template <typename TensorT, typename ContextT>
    struct Drop : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            stack.drop();
        }
        NWORD_CREATOR_DEFINE_LR(Drop)
    };

    template <typename TensorT, typename ContextT>
    struct Dup : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            stack.dup();
        }
        NWORD_CREATOR_DEFINE_LR(Dup)
    };

    template <typename TensorT, typename ContextT>
    struct Dup2 : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            stack.dup2();
        }
        NWORD_CREATOR_DEFINE_LR(Dup2)
    };

    template <typename TensorT, typename ContextT>
    struct Swap : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            stack.swap();
        }
        NWORD_CREATOR_DEFINE_LR(Swap)
    };

    template <typename TensorT, typename ContextT>
    struct Rot : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            stack.rot();
        }
        NWORD_CREATOR_DEFINE_LR(Rot)
    };

    template <typename TensorT, typename ContextT>
    struct Rev : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            int n = stack.pop_number();
            stack.rev(n);
        }
        NWORD_CREATOR_DEFINE_LR(Rev)
    };

    template <typename TensorT, typename ContextT>
    struct Add : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b + a );
        }
        NWORD_CREATOR_DEFINE_LR(Add)
    };

    template <typename TensorT, typename ContextT>
    struct Sub : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b - a );
        }
        NWORD_CREATOR_DEFINE_LR(Sub)
    };

    template <typename TensorT, typename ContextT>
    struct Mul : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b * a );
        }
        NWORD_CREATOR_DEFINE_LR(Mul)
    };

    template <typename TensorT, typename ContextT>
    struct Div : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b / a );
        }
        NWORD_CREATOR_DEFINE_LR(Div)
    };

    template <typename TensorT, typename ContextT>
    struct IntDiv : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( (int)(b / a) );
        }
        NWORD_CREATOR_DEFINE_LR(IntDiv)
    };

    template <typename TensorT, typename ContextT>
    struct Equals : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( a == b);
        }
        NWORD_CREATOR_DEFINE_LR(Equals)
    };

    template <typename TensorT, typename ContextT>
    struct Combin : public NativeWord<TensorT> {
        void run(Stack<TensorT>& stack) override {
            auto a = stack.pop_string();
            auto b = stack.pop_string();
            std::string c = b + a;
            stack.push_string( c );
        }
        NWORD_CREATOR_DEFINE_LR(Combin)
    };
}

template <typename TensorT, typename ContextT>
void Enviroment<TensorT, ContextT>::load_base_words() {
    // base words
    insert_native_word("drop", base::Drop<TensorT, ContextT>::creator );
    insert_native_word("dup", base::Dup<TensorT, ContextT>::creator );
    insert_native_word("dup2", base::Dup2<TensorT, ContextT>::creator );
    insert_native_word("swap", base::Swap<TensorT, ContextT>::creator );
    insert_native_word("rot", base::Rot<TensorT, ContextT>::creator );
    insert_native_word("rev", base::Rev<TensorT, ContextT>::creator );
    insert_native_word("?", base::Echo<TensorT, ContextT>::creator );
    insert_native_word("??", base::Dump<TensorT, ContextT>::creator );
    insert_native_word("^", base::Exit<TensorT, ContextT>::creator );

    insert_native_word("+", base::Add<TensorT, ContextT>::creator );
    insert_native_word("-", base::Sub<TensorT, ContextT>::creator );
    insert_native_word("*", base::Mul<TensorT, ContextT>::creator );
    insert_native_word("/", base::Div<TensorT, ContextT>::creator );
    insert_native_word("//", base::IntDiv<TensorT, ContextT>::creator );
    insert_native_word("==", base::Equals<TensorT, ContextT>::creator );

    insert_native_word("|", base::Combin<TensorT, ContextT>::creator );
}

template <typename TensorT, typename ContextT>
UserWord Enviroment<TensorT, ContextT>::compile(const std::string& txt) {
    struct _ {
        static bool parse_number(const std::string& token, double& value) {
            if (isdigit(token.at(0)) || (token.at(0) == '-' && token.length() >= 2 && isdigit(token.at(1)))) {
                if (token.find('.') != std::string::npos || token.find('e') != std::string::npos) { // double
                    value = atof(token.c_str());
                } else {
                    value = atol(token.c_str());
                }
                return true;
            }
            return false;
        }

        static void tokenize_line(std::string const &str_line, std::vector<std::string> &out) {
            std::string state = "SYMBOL";
            std::string current_str = "";

            for (size_t i = 0; i < str_line.size(); i++) {
                char cur = str_line[i];
                if ( state == "SYMBOL") {
                    if ( cur == ' ' || cur == '('  || cur == ')' || cur == '{' || cur == '}' ) {
                        // ending of a symbol
                        if (current_str != "") {
                            out.push_back(current_str);
                            current_str = "";
                        }
                        continue;
                    }
                    if ( cur == '[' || cur == ']' ) {
                        // ending of a symbol
                        if (current_str != "") {
                            out.push_back(current_str);
                            current_str = "";
                        }
                        out.push_back( std::string(1, cur) );
                        continue;
                    }

                    if ( cur == '"' ) {
                        if (current_str != "") {
                            vt_panic("tokenize_line error!");
                        }

                        state = "STRING";
                        current_str.push_back('"');
                        continue;
                    }
                    if ( cur == '\'' ) {
                        if (current_str != "") {
                            vt_panic("tokenize_line error!");
                        }

                        state = "STRING";
                        current_str.push_back('\'');
                        continue;
                    }

                    if ( cur == ';' ) {
                        if (current_str != "") {
                            out.push_back(current_str);
                        }
                        return;
                    }

                    current_str.push_back(cur);
                    continue;
                }
                if ( state == "STRING" ) {
                    if ( cur == '"' && current_str.at(0) == '"') {
                        current_str.push_back('"');
                        out.push_back(current_str);
                        current_str = "";
                        state = "SYMBOL";
                        continue;
                    }
                    if ( cur == '\'' && current_str.at(0) == '\'') {
                        current_str.push_back('\'');
                        out.push_back(current_str);
                        current_str = "";
                        state = "SYMBOL";
                        continue;
                    }
                    current_str.push_back(cur);
                }
            }
            if ( state == "STRING" ) {
                vt_panic("tokenize_line error, string must end in one line!");
            }
            if (current_str != "") {
                out.push_back(current_str);
            }
        }

        static bool is_valid_name(std::string const &str) {
            if ( str == "true" || str == "false" || str == "null" || str == "@" || str == "!" || str == "!!" || str == "jnz" || str =="jz" ) {
                return false;
            }
            if (str.find_first_not_of("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") == std::string::npos) {
                if ( str.find_first_of("0123456789") != 0 ) {
                    return true;
                }
            }
            return false;
        }
    };

    // 0. removed comments & tokenize
    std::vector<std::string> tokens;
    std::istringstream code_stream(txt);
    std::string line;
    while (std::getline(code_stream, line)) {
        _::tokenize_line(line,  tokens);
    }

    // 1. pre-processing code loop
    {
        bool block_comment = false;
        std::vector<std::string> new_tokens;

        // removed comments
        for (size_t i = 0; i < tokens.size(); i++) {
            auto token = tokens[i];
            if ( token == "/*" ) {
                if ( block_comment == true ) {
                    vt_panic("Find /* block comment nesting !");
                }
                block_comment = true;
                continue;
            }
            if ( token == "*/" ) {
                if ( block_comment == false ) {
                    vt_panic("Find */ block comment without /* begin !");
                }
                block_comment = false;
                continue;
            }
            if ( block_comment == true) {
                continue;
            }

            new_tokens.push_back(token);
        }
        tokens = new_tokens;
        new_tokens.clear();

        // extending loop
        bool in_loop = false;
        std::vector<std::string> loop_block;
        std::optional<int> loop_begin;
        std::optional<int> loop_end;
        for (size_t i = 0; i < tokens.size(); i++) {
            auto token = tokens[i];
            if ( token == "%for" ) {
                if ( in_loop == true ) {
                    vt_panic("Find loop block nesting !");
                }
                in_loop = true;
                continue;
            }
            if ( token == "%endf" ) {
                if ( in_loop == false ) {
                    vt_panic("Find %endf  without %for !");
                }
                if ( !loop_begin.has_value() || !loop_end.has_value() ) {
                    vt_panic("Can't find begin and end !");
                }
                if ( loop_end > loop_begin ) {
                    for (int l = loop_begin.value(); l <= loop_end.value(); l++) {
                        std::stringstream ss;
                        ss << l;
                        std::string dst = ss.str();
                        for (size_t j = 0; j < loop_block.size(); j++) {
                            std::string new_str = std::regex_replace( loop_block[j], std::regex("%%"), dst);
                            new_tokens.push_back(new_str);
                        }
                    }
                } else {
                   for (int l = loop_begin.value(); l >= loop_end.value(); l--) {
                        std::stringstream ss;
                        ss << l;
                        std::string dst = ss.str();
                        for (size_t j = 0; j < loop_block.size(); j++) {
                            std::string new_str = std::regex_replace( loop_block[j], std::regex("%%"), dst);
                            new_tokens.push_back(new_str);
                        }
                   }
                }
                loop_block.clear();
                loop_begin.reset();
                loop_end.reset();
                in_loop = false;
                continue;
            }
            if ( in_loop == true ) {
                double num;
                if ( !loop_begin.has_value() ) {
                    if ( !_::parse_number(token, num) ) {
                        vt_panic("can't find loop begin number");
                    }
                    loop_begin = (int)num;
                    continue;
                }
                if ( !loop_end.has_value() ) {
                    if ( !_::parse_number(token, num) ) {
                        vt_panic("can't find loop begin number");
                    }

                    loop_end = (int)num;
                    continue;
                }
                loop_block.push_back(token);
                continue;
            }
            new_tokens.push_back(token);
        }
        tokens = new_tokens;
    }

    UserWord main_code;
    std::optional<UserWord> user_code;
    std::optional<size_t> list_count;
    list_count.reset();

    // 2. processing main code
    for (size_t i = 0; i < tokens.size(); i++) {
        auto token = tokens[i];
        // first pass, processing command primitive
        if ( token == "%def" ) {
            if ( user_code.has_value() ) {
                vt_panic("Can't define a new user word inside another user word!");
            }
            if ( list_count.has_value() ) {
                vt_panic("Can't define a new user word inside a list macro!");
            }
            user_code = UserWord();

            i = i + 1;
            if ( i >= tokens.size() ) {
                vt_panic("Can't find #end for #def!");
            }
            token = tokens[i];
            if ( _::is_valid_name(token) ) {
                if ( user_words_.find(token) == user_words_.end() ) {
                    if ( native_words_.find(token) == native_words_.end() ) {
                        user_code.value().push_back( WordCode::new_string( token) );
                        continue;
                    }
                }
            }
            vt_panic("Can't a valid name for #def macro!");

        } else if ( token == "[" ) {
            list_count = 0;
            continue;
        } else if ( token == "%end" ) {
            if ( list_count.has_value() ) {
                vt_panic("Can't ending a word or in a list macro.");
            }

            if ( user_code.has_value() ) {
                auto w = user_code.value();
                if ( w.size() < 1 ) {
                    vt_panic("define macro error, must including a word name!");
                }
                vt_assert( w[0].type_ == WordCode::String, "first of define macro, must be a name!");

                auto name = w[0].str_;
                w.erase(w.begin());
                user_words_[name] = w;

                user_code.reset();
                continue;
            }

            vt_panic("Find #end without #def");
        } else if ( token == "]" ) {
            if ( !list_count.has_value() ) {
                vt_panic("']' list macro appears without begin '['!");
            }

            UserWord* target = &main_code;
            if ( user_code.has_value() ) {
                target = &user_code.value();
            }

            double lcount = list_count.value();
            target->push_back( WordCode::new_number( lcount ) );

            list_count.reset();
            continue;
        }
        // second pass code to number, string,builtin, native, or user
        WordCode newCode;

        if ( token == "true" ) {
            newCode = WordCode::new_number( 1.0 );
        } else if ( token == "false") {
            newCode = WordCode::new_number( 0.0 );
        } else if ( token == "null" ) {
            newCode = WordCode::new_string( "" );
        } else if ( token == "@" ||
                    token == "!"  ||
                    token == "!!" ||
                    token == "jnz"||
                    token == "jz" ) {
            newCode = WordCode::new_builtin( token );
        } else if ( token[0] == '"' || token[0] == '\'' || token[0] == '$' ) {
            if ( token[0] == '"' || token[0] == '\'' ) {
                vt_assert( token.size() >= 2, " string must begin \" ' or $");
                vt_assert( token[0] == token.back() , " string length wrong");
                token = token.substr(1, token.size() - 2);
            }
            newCode = WordCode::new_string( token );
        } else if ( _::parse_number(token, newCode.num_) ) {
            newCode = WordCode::new_number( newCode.num_ );
        } else if ( native_words_.find( token ) != native_words_.end() ) {
            newCode = WordCode::new_native( token );
        } else if ( user_words_.find( token ) != user_words_.end() ) {
            newCode = WordCode::new_user( token );
        } else {
            std::cerr << token << std::endl;
            vt_panic("Find an invalid symbol is not string, number, builtin, user or native!");
        }

        UserWord* target = &main_code;
        if ( user_code.has_value() ) {
            target = &user_code.value();
        }
        if ( list_count.has_value() ) {
            list_count = list_count.value() + 1;
        }

        target->push_back(newCode);
    }

    if (list_count.has_value()) {
        vt_panic("List macro without ']' ending!");
    }
    if (user_code.has_value()) {
        vt_panic("Define macro without #end ending!");
    }

    return main_code;
}

} // end of namespace
#endif
