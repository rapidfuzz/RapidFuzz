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

    SECTION( "weighted levenshtein calculates correct ratios" ) {
        REQUIRE( levenshtein::normalized_weighted_distance("test", "test") == 1.0 );
        REQUIRE( levenshtein::normalized_weighted_distance("test", "tes") == Approx(0.857).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance("te", "et") == Approx(0.5).epsilon(0.01) ); 
        REQUIRE( levenshtein::normalized_weighted_distance("test", "tess") == Approx(0.75).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance("test", "xxxx") == 0.0 );
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

    SECTION( "weighted levenshtein calculates correct ratio") {
        REQUIRE( levenshtein::normalized_weighted_distance(test, test) == 1.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, insert) == Approx(0.93).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(test, insert_delete) == Approx(0.875).epsilon(0.01) ); 
        REQUIRE( levenshtein::normalized_weighted_distance(test, replace) == Approx(0.875).epsilon(0.01) );
        REQUIRE( levenshtein::normalized_weighted_distance(test, replace_all) == 0.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, combined) == 1.0 );
        REQUIRE( levenshtein::normalized_weighted_distance(test, combined, " ") == Approx(0.94).epsilon(0.01) );
    }
}