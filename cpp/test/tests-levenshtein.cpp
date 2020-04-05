#include "catch2/catch.hpp"
#include <string_view>
#include <vector>
#include <algorithm>
#include <boost/utility/string_view.hpp>

#include "../src/levenshtein.hpp"

TEST_CASE( "levenshtein works with string_views", "[string_view]" ) {
    boost::string_view test = "aaaa";
    boost::string_view no_suffix = "aaa";
    boost::string_view no_suffix2 = "aaab";
    boost::string_view swapped1 = "abaa";
    boost::string_view swapped2 = "baaa";
    boost::string_view replace_all = "bbbb";

    SECTION( "weighted levenshtein calculates correct distances" ) {
        REQUIRE( levenshtein::weighted_distance(test, test) == 0 );
        REQUIRE( levenshtein::weighted_distance(test, no_suffix) == 1 );
        REQUIRE( levenshtein::weighted_distance(swapped1, swapped2) == 2 ); 
        REQUIRE( levenshtein::weighted_distance(test, no_suffix2) == 2 );
        REQUIRE( levenshtein::weighted_distance(test, replace_all) == 8 );
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
            {4, std::vector<std::size_t>{0}, 1, 1});
        
        matrix_cmp(
            levenshtein::matrix(test, no_suffix),
            {3, std::vector<std::size_t>{0, 1}, 2, 1});
        
        matrix_cmp(
            levenshtein::matrix(swapped1, swapped2),
            {0, std::vector<std::size_t>{ 0, 1, 2, 1, 1, 1, 2, 1, 2 }, 3, 3});
        
        matrix_cmp(
            levenshtein::matrix(test, no_suffix2),
            {3, std::vector<std::size_t>{0, 1, 1, 1}, 2, 2});
        
        matrix_cmp(
            levenshtein::matrix(test, replace_all),
            {0, std::vector<std::size_t>{0, 1, 2, 3, 4, 1, 1, 2, 3, 4, 2, 2, 2, 3, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4}, 5, 5});
    }

    SECTION( "levenshtein calculates correct levenshtein matrix" ) {
        auto matching_block_cmp = [](std::vector<levenshtein::MatchingBlock> res, std::vector<levenshtein::MatchingBlock> check) {
            auto res_iter = res.begin();
            auto check_iter = check.begin();
            while (res_iter != res.end() && check_iter != check.end()) {
                REQUIRE(res_iter->len == check_iter->len);
                REQUIRE(res_iter->first_start == check_iter->first_start);
                REQUIRE(res_iter->first_start == check_iter->first_start);
                ++res_iter;
                ++check_iter;
            }
            REQUIRE(res_iter == res.end());
            REQUIRE(check_iter == check.end());
        };
        
        std::vector<levenshtein::MatchingBlock> result;
        result.emplace_back(4,4,0);
        matching_block_cmp( levenshtein::matching_blocks(test, replace_all), result);
    }

    struct MatchingBlock {
    std::size_t first_start;
    std::size_t second_start;
    std::size_t len;
    MatchingBlock(std::size_t first_start, std::size_t second_start, std::size_t len)
        : first_start(first_start)
        , second_start(second_start)
        , len(len)
    {}
};

}