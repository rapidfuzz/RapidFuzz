#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../src/levenshtein.hpp"

// handle a lot more cases e.g. with common prefix/suffix for each test case
TEST_CASE( "strings can can be used can be used", "[vector<int>]" ) {
    SECTION( "levenshtein" ) {
    }
    SECTION( "weighted levenshtein" ) {
    }
    SECTION( "normalized levenshtein" ) {
    }
    SECTION( "normalized weighted levenshtein" ) {
    }
}

TEST_CASE( "strings with string_views can can be used can be used", "[vector<int>]" ) {
    SECTION( "levenshtein" ) {
    }
    SECTION( "weighted levenshtein" ) {
    }
    SECTION( "normalized levenshtein" ) {
    }
    SECTION( "normalized weighted levenshtein" ) {
    }
}

TEST_CASE( "vectors can be used can be used", "[vector<int>]" ) {
    SECTION( "levenshtein" ) {
    }
    SECTION( "weighted levenshtein" ) {
    }
    SECTION( "normalized levenshtein" ) {
    }
    SECTION( "normalized weighted levenshtein" ) {
    }
}

TEST_CASE( "vectors of strings can be used can be used", "[vector<string>]" ) {
    SECTION( "levenshtein" ) {
    }
    SECTION( "weighted levenshtein" ) {
    }
    SECTION( "normalized levenshtein" ) {
    }
    SECTION( "normalized weighted levenshtein" ) {
    }
}

TEST_CASE( "vectors of vectors can be used can be used", "[vector<vector<int>>]" ) {
    SECTION( "levenshtein" ) {
    }
    SECTION( "weighted levenshtein" ) {
    }
    SECTION( "normalized levenshtein" ) {
    }
    SECTION( "normalized weighted levenshtein" ) {
    }
}
