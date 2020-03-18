#include "catch2/catch.hpp"
#include <string_view>
#include <vector>
#include <algorithm>

#include "../src/levenshtein.hpp"

TEST_CASE( "levenshtein works with string_views", "[string_view]" ) {
    std::string_view test = "aaaa";
    std::string_view no_suffix = "aaa";
    std::string_view no_suffix2 = "aaab";
    std::string_view swapped1 = "abaa";
    std::string_view swapped2 = "baaa";
    std::string_view replace_all = "bbbb";

    SECTION( "weighted levenshtein calculates correct distances" ) {
        REQUIRE( levenshtein::weighted_distance(test, test) == 0 );
        REQUIRE( levenshtein::weighted_distance(test, no_suffix) == 1 );
        REQUIRE( levenshtein::weighted_distance(swapped1, swapped2) == 2 ); 
        REQUIRE( levenshtein::weighted_distance(test, no_suffix2) == 2 );
        REQUIRE( levenshtein::weighted_distance(test, replace_all) == 8 );
        REQUIRE( levenshtein::weighted_distance(test, replace_all, 3) == std::numeric_limits<size_t>::max() );
    }

    SECTION( "weighted levenshtein calculates correct ratios" ) {
        REQUIRE( levenshtein::normalized_weighted_distance(test, test) == 1.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, no_suffix) == Approx(0.857).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(swapped1, swapped2) == Approx(0.75).epsilon(0.01) ); 
        REQUIRE( levenshtein::normalized_weighted_distance(test, no_suffix2) == Approx(0.75).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(test, replace_all) == 0.0 );
    }


    SECTION( "levenshtein calculates correct levenshtein matrix" ) {
        auto matrix_cmp = [](levenshtein::Matrix a, levenshtein::Matrix b) {
            REQUIRE( a.prefix_len == b.prefix_len);
            REQUIRE( a.matrix == b.matrix);
            REQUIRE( a.matrix_columns == b.matrix_columns);
            REQUIRE( a.matrix_rows == b.matrix_rows);
        };


        matrix_cmp(
            levenshtein::matrix(test, test),
            {4, std::vector<size_t>{0}, 1, 1});
        
        matrix_cmp(
            levenshtein::matrix(test, no_suffix),
            {3, std::vector<size_t>{0, 1}, 2, 1});
        
        matrix_cmp(
            levenshtein::matrix(swapped1, swapped2),
            {0, std::vector<size_t>{ 0, 1, 2, 1, 1, 1, 2, 1, 2 }, 3, 3});
        
        matrix_cmp(
            levenshtein::matrix(test, no_suffix2),
            {3, std::vector<size_t>{0, 1, 1, 1}, 2, 2});
        
        matrix_cmp(
            levenshtein::matrix(test, replace_all),
            {0, std::vector<size_t>{0, 1, 2, 3, 4, 1, 1, 2, 3, 4, 2, 2, 2, 3, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4}, 5, 5});
    }

    SECTION( "levenshtein calculates correct levenshtein editops" ) {
        auto edit_op_cmp = [](std::vector<levenshtein::EditOp> res, std::vector<levenshtein::EditOp> check) {
            auto res_iter = res.begin();
            auto check_iter = check.begin();
            while (res_iter != res.end() && check_iter != check.end()) {
                REQUIRE(res_iter->op_type == check_iter->op_type);
                REQUIRE(res_iter->first_start == check_iter->first_start);
                REQUIRE(res_iter->first_start == check_iter->first_start);
                ++res_iter;
                ++check_iter;
            }
            REQUIRE(res_iter == res.end());
            REQUIRE(check_iter == check.end());
        };

        auto ed_replace = [](size_t pos1, size_t pos2) {
            return levenshtein::EditOp{levenshtein::EditType::EditReplace, pos1, pos2};
        };

        auto ed_delete = [](size_t pos1, size_t pos2) {
            return levenshtein::EditOp{levenshtein::EditType::EditDelete, pos1, pos2};
        };

        auto ed_insert = [](size_t pos1, size_t pos2) {
            return levenshtein::EditOp{levenshtein::EditType::EditInsert, pos1, pos2};
        };


        edit_op_cmp(levenshtein::editops(test, test), {});

        edit_op_cmp(
            levenshtein::editops(test, no_suffix),
            { ed_delete(3, 3) });

        edit_op_cmp(
            levenshtein::editops(no_suffix, test),
            { ed_insert(3, 3) });

        edit_op_cmp(
            levenshtein::editops(swapped1, swapped2),
            { ed_replace(0, 0), ed_replace(1, 1) });

        edit_op_cmp(
            levenshtein::editops(test, no_suffix2),
            { ed_replace(3, 3) });

        edit_op_cmp(
            levenshtein::editops(test, replace_all),
            { ed_replace(0, 0), ed_replace(1, 1), ed_replace(2, 2), ed_replace(3, 3) });
    }
}

TEST_CASE( "levenshtein works with vectors of string_views", "[vector<string_view>]" ) {
    std::vector<std::string_view> test {"test", "test"};
    std::vector<std::string_view> combined {"testtest"};
    std::vector<std::string_view> insert {"tes", "test"};
    std::vector<std::string_view> replace {"test", "tess"};
    std::vector<std::string_view> replace_all {"xxxx", "xxxx"};
    std::vector<std::string_view> insert_delete {"etst", "test"};

    SECTION( "weighted levenshtein calculates correct distances") {
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