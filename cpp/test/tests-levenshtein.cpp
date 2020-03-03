#include "catch2/catch.hpp"
#include <string_view>

#include "../src/levenshtein.hpp"

TEST_CASE( "levenshtein works with string_views", "[vector<int>]" ) {
    SECTION( "weighted levenshtein works correctly" ) {
        
        REQUIRE( levenshtein::weighted_distance("test", "test") == 0 );
        REQUIRE( levenshtein::weighted_distance("test", "tes") == 1 );
        REQUIRE( levenshtein::weighted_distance("te", "et") == 2 ); 
        REQUIRE( levenshtein::weighted_distance("test", "tess") == 2 );
        REQUIRE( levenshtein::weighted_distance("test", "xxxx") == 8 );
    }
}