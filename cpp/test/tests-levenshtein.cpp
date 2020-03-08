#include "catch2/catch.hpp"
#include <string_view>
#include <vector>
#include <algorithm>

#include "../src/levenshtein.hpp"

TEST_CASE( "levenshtein works with string_views", "[string_view]" ) {
    std::string_view test = "test";
    std::string_view no_suffix = "tes";
    std::string_view no_suffix2 = "tess";
    std::string_view swapped = "etst";
    std::string_view replace_all = "xxxx";

    SECTION( "weighted levenshtein calculates correct distances" ) {
        REQUIRE( levenshtein::weighted_distance(test, test) == 0 );
        REQUIRE( levenshtein::weighted_distance(test, no_suffix) == 1 );
        REQUIRE( levenshtein::weighted_distance(test, swapped) == 2 ); 
        REQUIRE( levenshtein::weighted_distance(test, no_suffix2) == 2 );
        REQUIRE( levenshtein::weighted_distance(test, replace_all) == 8 );
        REQUIRE( levenshtein::weighted_distance(test, replace_all, 3) == std::numeric_limits<size_t>::max() );
    }

    SECTION( "weighted levenshtein calculates correct ratios" ) {
        REQUIRE( levenshtein::normalized_weighted_distance(test, test) == 1.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, no_suffix) == Approx(0.857).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(test, swapped) == Approx(0.75).epsilon(0.01) ); 
        REQUIRE( levenshtein::normalized_weighted_distance(test, no_suffix2) == Approx(0.75).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(test, replace_all) == 0.0 );
    }


    SECTION( "levenshtein calculates correct levenshtein matrix" ) {
        auto result = levenshtein::matrix("test", "test");
        REQUIRE( result.prefix_len == 4);
        REQUIRE( result.matrix == std::vector<size_t>{0});
        REQUIRE( result.matrix_columns == 1);
        REQUIRE( result.matrix_rows == 1);

        result = levenshtein::matrix("test", "tes");
        REQUIRE( result.prefix_len == 3);
        REQUIRE( result.matrix == std::vector<size_t>{0,1});
        REQUIRE( result.matrix_columns == 2);
        REQUIRE( result.matrix_rows == 1);

        result = levenshtein::matrix("te", "et");
        REQUIRE( result.prefix_len == 0);
        REQUIRE( result.matrix == std::vector<size_t>{ 0, 1, 2, 1, 1, 1, 2, 1, 2 });
        REQUIRE( result.matrix_columns == 3);
        REQUIRE( result.matrix_rows == 3);

        result = levenshtein::matrix("test", "tess");
        REQUIRE( result.prefix_len == 3);
        REQUIRE( result.matrix == std::vector<size_t>{0, 1, 1, 1});
        REQUIRE( result.matrix_columns == 2);
        REQUIRE( result.matrix_rows == 2);

        result = levenshtein::matrix("test", "xxxx");
        REQUIRE( result.prefix_len == 0);
        REQUIRE( result.matrix == std::vector<size_t>{0, 1, 2, 3, 4, 1, 1, 2, 3, 4, 2, 2, 2, 3, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4});
        REQUIRE( result.matrix_columns == 5);
        REQUIRE( result.matrix_rows == 5);
    }

    SECTION( "levenshtein calculates correct levenshtein editops" ) {
        auto edit_op_comp = [](auto res, auto check) {
            bool result = true;
            auto res_iter = res.begin();
            auto check_iter = check.begin();
            while (res_iter != res.end() && check_iter != check.end()) {
                result &= res_iter->op_type == check_iter->op_type;
                result &= res_iter->first_start == check_iter->first_start;
                result &= res_iter->second_start == check_iter->second_start;
                ++res_iter;
                ++check_iter;
            }
            result &= res_iter == res.end();
            result &= check_iter == check.end();
            return result;
        };


        {
            auto result = levenshtein::editops("test", "test");
            REQUIRE(result.empty());
        }
        {
            auto result = levenshtein::editops("test", "tes");
            levenshtein::EditOp edit_op{ levenshtein::EditType::EditDelete, 3, 3 };
            std::vector<levenshtein::EditOp> check_result { edit_op };
            REQUIRE(edit_op_comp(result, check_result));
        }
        {
            auto result = levenshtein::editops("te", "et");
            levenshtein::EditOp edit_op1{ levenshtein::EditType::EditReplace, 0, 0};
            levenshtein::EditOp edit_op2{ levenshtein::EditType::EditReplace, 1, 1};
            std::vector<levenshtein::EditOp> check_result { edit_op1, edit_op2 };
            REQUIRE(edit_op_comp(result, check_result));
        }
        {
            auto result = levenshtein::editops("test", "tess");
            levenshtein::EditOp edit_op{ levenshtein::EditType::EditReplace, 3, 3};
            std::vector<levenshtein::EditOp> check_result { edit_op };
            REQUIRE(edit_op_comp(result, check_result));
        }
        {
            auto result = levenshtein::editops("test", "xxxx");
            levenshtein::EditOp edit_op1{ levenshtein::EditType::EditReplace, 0, 0};
            levenshtein::EditOp edit_op2{ levenshtein::EditType::EditReplace, 1, 1};
            levenshtein::EditOp edit_op3{ levenshtein::EditType::EditReplace, 2, 2};
            levenshtein::EditOp edit_op4{ levenshtein::EditType::EditReplace, 3, 3};
            std::vector<levenshtein::EditOp> check_result { edit_op1, edit_op2, edit_op3, edit_op4 };
            REQUIRE(edit_op_comp(result, check_result));
        }
    }
}

TEST_CASE( "levenshtein works with vectors of string_views", "[vector<string_view>]" ) {
    std::vector<std::string_view> test {"test", "test"};
    std::vector<std::string_view> no_affix {"xest", "tesx"};
    std::vector<std::string_view> combined {"testtest"};
    std::vector<std::string_view> insert {"tes", "test"};
    std::vector<std::string_view> replace {"test", "tess"};
    std::vector<std::string_view> replace_all {"xxxx", "xxxx"};
    std::vector<std::string_view> insert_delete {"etst", "test"};

    SECTION( "weighted levenshtein calculates correct distances") {
        levenshtein::weighted_distance(test, no_affix);
        REQUIRE( levenshtein::weighted_distance(test, test) == 0 );
        REQUIRE( levenshtein::weighted_distance(test, insert) == 1 );
        REQUIRE( levenshtein::weighted_distance(test, insert_delete) == 2 ); 
        REQUIRE( levenshtein::weighted_distance(test, replace) == 2 );
        REQUIRE( levenshtein::weighted_distance(test, replace_all) == 16 );
        REQUIRE( levenshtein::weighted_distance(test, replace_all, 7) == std::numeric_limits<size_t>::max() );
        REQUIRE( levenshtein::weighted_distance(test, combined) == 0 );
        REQUIRE( levenshtein::weighted_distance(test, combined, 0.0, " ") == 1 );
    }

    SECTION( "weighted levenshtein calculates correct ratio") {
        REQUIRE( levenshtein::normalized_weighted_distance(test, test) == 1.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, insert) == Approx(0.93).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(test, insert_delete) == Approx(0.875).epsilon(0.01) ); 
        REQUIRE( levenshtein::normalized_weighted_distance(test, replace) == Approx(0.875).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(test, replace_all) == 0.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, combined) == 1.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, combined, 0.0, " ") == Approx(0.94).epsilon(0.01) );
    }
}