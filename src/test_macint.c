#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#include "type_interact.h"

int main(int argc, const char** argv) {
	size_t i, j;
	size_t num_tries = 200000;
	char macs[num_tries][18];
	uint64_t sscanf_res[num_tries];
	uint64_t htoi_res[num_tries];
	unsigned char arr[6];
	double tstart, tend, t;

	struct timespec start, end;

	srand(time(NULL));

	for (i = 0; i < num_tries; i++) {
		for (j = 0; j < 6; j++)
			arr[j] = rand();

		sprintf(macs[i], "%02x:%02x:%02x:%02x:%02x:%02x", arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
	}

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	for (i = 0; i < num_tries; i++)
		sscanf_res[i] = mac_addr_to_num_sscanf(macs[i]);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	tstart = start.tv_sec * 1e6 + start.tv_nsec * 1e-3;
	tend = end.tv_sec * 1e6 + end.tv_nsec * 1e-3;
	t = (tend - tstart) / num_tries;

	printf("sscanf: %.4lf us per addr\n", t);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	for (i = 0; i < num_tries; i++)
		htoi_res[i] = mac_addr_to_num_htoi(macs[i]);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	tstart = start.tv_sec * 1e6 + start.tv_nsec * 1e-3;
	tend = end.tv_sec * 1e6 + end.tv_nsec * 1e-3;
	t = (tend - tstart) / num_tries;

	printf("htoi: %.4lf us per addr\n", t);

	for (i = 0; i < num_tries; i++) {
		if (sscanf_res[i] != htoi_res[i])
			printf("Failure at %s: sscanf %lu, htoi %lu\n", macs[i], sscanf_res[i], htoi_res[i]);
	}
}
