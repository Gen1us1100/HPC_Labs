// merge_sort.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void merge(int *arr, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L); free(R);
}

void mergeSortSeq(int *arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSortSeq(arr, l, m);
        mergeSortSeq(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}
#define THRESHOLD 5000  // Can tune this based on experimentation

void parallelMergeSort(int arr[], int left, int right) {
    if (left < right) {
        if ((right - left) < THRESHOLD) {
            mergeSortSeq(arr, left, right); // fall back to sequential
            return;
        }

        int mid = (left + right) / 2;

        #pragma omp task
        parallelMergeSort(arr, left, mid);

        #pragma omp task
        parallelMergeSort(arr, mid + 1, right);

        #pragma omp taskwait
        merge(arr, left, mid, right);
    }
}

/*void mergeSortPar(int *arr, int l, int r, int depth) {
    if (l < r) {
        if (depth <= 0) {
            mergeSortSeq(arr, l, r);
            return;
        }
        int m = l + (r - l) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortPar(arr, l, m, depth - 1);

            #pragma omp section
            mergeSortPar(arr, m + 1, r, depth - 1);
        }
        merge(arr, l, m, r);
    }
}*/

void generateRandomArray(int *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = rand() % 10000;
}

void printArray(int *arr, int n) {
    for (int i = 0; i < n && i < 20; i++)
        printf("%5d ", arr[i]);
    if (n > 20) printf("...");
    printf("\n");
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);
    int *arr1 = malloc(n * sizeof(int));
    int *arr2 = malloc(n * sizeof(int));
    srand(time(NULL));
    generateRandomArray(arr1, n);
    for (int i = 0; i < n; i++) arr2[i] = arr1[i];

    printf("\nOriginal array:\n");
    printArray(arr1, n);

    double start, end;

    start = omp_get_wtime();
    mergeSortSeq(arr1, 0, n - 1);
    end = omp_get_wtime();
    double seqTime = end - start;

    start = omp_get_wtime();
    if (n < THRESHOLD) {
	parallelMergeSort(arr2, 0, n - 1);  // Will run sequential anyway
    } else {
    	#pragma omp parallel
        {
	#pragma omp single
	parallelMergeSort(arr2, 0, n - 1);
        }
    }


    end = omp_get_wtime();
    double parTime = end - start;

    printf("\nSorted (Sequential):\n");
    printArray(arr1, n);
    printf("Time (Sequential Merge Sort): %.6f s\n", seqTime);

    printf("\nSorted (Parallel):\n");
    printArray(arr2, n);
    printf("Time (Parallel Merge Sort):   %.6f s\n", parTime);

    printf("\n%-25s %.2fx speedup\n", "Speedup (Par / Seq):", seqTime / parTime);

    free(arr1);
    free(arr2);
    return 0;
}

