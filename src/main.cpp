#include <cstdio>
#include <omp.h>

using namespace std;

int main()
{
    printf("OpenMP max threads: %d\n", omp_get_max_threads());

    #pragma omp parallel
    {
        #pragma omp critical
        printf("    hello from thread %d\n", omp_get_thread_num());
    }
    return 0;
}