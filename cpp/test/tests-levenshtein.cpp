#include "catch2/catch.hpp"
#include <string_view>
#include <vector>

#include "../src/levenshtein.hpp"

TEST_CASE( "levenshtein works with string_views", "[string_view]" ) {
    SECTION( "weighted levenshtein calculates correct distances" ) {
        
        REQUIRE( levenshtein::weighted_distance("test", "test") == 0 );
        REQUIRE( levenshtein::weighted_distance("test", "tes") == 1 );
        REQUIRE( levenshtein::weighted_distance("te", "et") == 2 ); 
        REQUIRE( levenshtein::weighted_distance("test", "tess") == 2 );
        REQUIRE( levenshtein::weighted_distance("test", "xxxx") == 8 );
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
        REQUIRE( levenshtein::weighted_distance(test, combined) == 0 );
        REQUIRE( levenshtein::weighted_distance(test, combined, " ") == 1 );
    }
}