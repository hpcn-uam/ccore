
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>

#define SEC_TO_NSEC 1000000000
#define TS_GET_SECS(x)  (x / SEC_TO_NSEC)
#define TS_GET_NSECS(x) (x % SEC_TO_NSEC)

#define STRINGIFY(s) #s

#define RUN_TEST(f) {fprintf(stderr, "%s: ", STRINGIFY(f)); f(); fprintf(stderr, "PASSED!\n");}

static void read_unixtstamp(char* word, uint64_t* dest)
{
	char *ptr;
	uint64_t base = SEC_TO_NSEC;

	/* NOTE: negative timestamps are converted to -1 */
	if (*word == '-') {
		*dest = -1;
		return ;
	}

	errno = 0;
	*dest = strtol(word, &ptr, 10);

	if (errno == ERANGE) {
		fprintf(stderr, "ccore WARNING: overflow parsing timestamp (%.*s)!\n", (int)(ptr-word + 1), word);
	}

	/*
	 A unix timestamp consists in a number from 0 to 2**32-1, so it
	 can be stored in 32 bits. A unix timestamp in nanoseconds,
	 has upto 20 digits, so it can be stored in 64 bits.
	 */

	*dest *= base;
	base = SEC_TO_NSEC / 10;
	/* ptr should point to the dot, skip it!
	   if it points to something else, stop parsing
	   note: we can't use strtol here because
	         the number might not be 0 padded
	 */
	if (*ptr == '.') {
		while (isdigit(*++ptr) && base > 0) {
			*dest += (uint64_t)(*ptr - '0') * base;
			base /= 10;
		}
	}
}

static void test_nine_decimals()
{
	char *str = "1234.000000001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 1 );
}

static void test_eight_decimals()
{
	char *str = "1234.00000001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 10 );
}

static void test_seven_decimals()
{
	char *str = "1234.0000001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 100 );
}

static void test_six_decimals()
{
	char *str = "1234.000001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 1000 );
}

static void test_five_decimals()
{
	char *str = "1234.00001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 10000 );
}

static void test_four_decimals()
{
	char *str = "1234.0001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 100000 );
}

static void test_three_decimals()
{
	char *str = "1234.001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 1000000 );
}

static void test_two_decimals()
{
	char *str = "1234.01";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 10000000 );
}

static void test_one_decimal()
{
	char *str = "1234.1";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 100000000 );
}

static void test_zero_padding()
{
	char *str = "1234.1000";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 100000000 );
}

static void test_no_decimals_with_dot()
{
	char *str = "1234.";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 0 );
}

static void test_no_decimals_without_dot()
{
	char *str = "1234";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 0 );
}

static void test_no_timestamp()
{
	char *str = " ";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 0 &&
	        TS_GET_NSECS(dest) == 0 );
}

static void test_empty()
{
	char *str = "";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 0 &&
	        TS_GET_NSECS(dest) == 0 );
}

static void test_minus_one()
{
	char *str = "-1";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( dest == (uint64_t)-1);
}

static void test_invalid()
{
	char *str = "asd";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 0 &&
	        TS_GET_NSECS(dest) == 0 );
}

static void test_too_long()
{
	char *str = "1234.0123456789";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 12345678 );
}

static void test_too_long_zeros()
{
	char *str = "1234.0000000001";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1234 &&
	        TS_GET_NSECS(dest) == 0 );
}

static void test_real_timestamp()
{
	char *str = "1446253798.912215022";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1446253798 &&
	        TS_GET_NSECS(dest) == 912215022 );
}

static void test_dont_continue_past_space()
{
	char *str = "1262325600 1262325660";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( TS_GET_SECS(dest) == 1262325600 &&
	        TS_GET_NSECS(dest) == 0 );
}

static void test_negative()
{
	char *str = "-1234.0123456789";
	uint64_t dest;
	read_unixtstamp(str, &dest);
	assert( dest == (uint64_t)-1 );
}


int main(int argc, char const *argv[])
{
	(void)argc;
	(void)argv;

	RUN_TEST(test_nine_decimals);
	RUN_TEST(test_eight_decimals);
	RUN_TEST(test_seven_decimals);
	RUN_TEST(test_six_decimals);
	RUN_TEST(test_five_decimals);
	RUN_TEST(test_four_decimals);
	RUN_TEST(test_three_decimals);
	RUN_TEST(test_two_decimals);
	RUN_TEST(test_one_decimal);
	RUN_TEST(test_zero_padding);
	RUN_TEST(test_no_decimals_with_dot);
	RUN_TEST(test_no_decimals_without_dot);
	RUN_TEST(test_no_timestamp);
	RUN_TEST(test_empty);
	RUN_TEST(test_minus_one);
	RUN_TEST(test_invalid);
	RUN_TEST(test_too_long);
	RUN_TEST(test_too_long_zeros);
	RUN_TEST(test_real_timestamp);
	RUN_TEST(test_dont_continue_past_space);
	RUN_TEST(test_negative);
	fprintf(stderr, "ALL TESTS PASSED!\n");
	return 0;
}

