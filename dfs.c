#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

// use following command to instruct linux for unlimited stack space and avoid segmentation fault
// ulimit -s unlimited
typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node** createGraph(int n) {
    Node** adjList = (Node**)malloc(n * sizeof(Node*));
    for (int i = 0; i < n; i++) adjList[i] = NULL;
    return adjList;
}

void addEdge(Node** adjList, int u, int v) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = adjList[u];
    adjList[u] = newNode;

    newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = u;
    newNode->next = adjList[v];
    adjList[v] = newNode;
}

void generateRandomGraph(Node** adjList, int n) {
    srand(time(NULL));
    for (int i = 1; i < n; i++) {
        int v = rand() % i;
        addEdge(adjList, i, v);
    }

    int extraEdges = n;
    for (int i = 0; i < extraEdges; i++) {
        int u = rand() % n;
        int v = rand() % n;
        if (u != v) addEdge(adjList, u, v);
    }
}

void dfsSequentialUtil(Node** adjList, int v, bool* visited) {
    visited[v] = true;
    for (Node* temp = adjList[v]; temp; temp = temp->next) {
        if (!visited[temp->vertex]) {
            dfsSequentialUtil(adjList, temp->vertex, visited);
        }
    }
}

void dfsSequential(Node** adjList, int n, int start) {
    bool* visited = (bool*)calloc(n, sizeof(bool));
    dfsSequentialUtil(adjList, start, visited);
    free(visited);
}

void dfsParallelUtil(Node** adjList, int v, bool* visited, omp_lock_t* locks) {
    omp_set_lock(&locks[v]);
    if (visited[v]) {
        omp_unset_lock(&locks[v]);
        return;
    }
    visited[v] = true;
    omp_unset_lock(&locks[v]);

    // Count neighbors first
    int neighborCount = 0;
    for (Node* temp = adjList[v]; temp; temp = temp->next) neighborCount++;

    int* neighbors = (int*)malloc(neighborCount * sizeof(int));
    int i = 0;
    for (Node* temp = adjList[v]; temp; temp = temp->next)
        neighbors[i++] = temp->vertex;

    #pragma omp parallel for
    for (int j = 0; j < neighborCount; j++) {
        int u = neighbors[j];
        #pragma omp task firstprivate(u)
        {
            dfsParallelUtil(adjList, u, visited, locks);
        }
    }

    #pragma omp taskwait
    free(neighbors);
}

void dfsParallel(Node** adjList, int n, int start) {
    bool* visited = (bool*)calloc(n, sizeof(bool));
    omp_lock_t* locks = (omp_lock_t*)malloc(n * sizeof(omp_lock_t));
    for (int i = 0; i < n; i++) omp_init_lock(&locks[i]);

    #pragma omp parallel
    {
        #pragma omp single
        dfsParallelUtil(adjList, start, visited, locks);
    }

    for (int i = 0; i < n; i++) omp_destroy_lock(&locks[i]);
    free(locks);
    free(visited);
}

int main() {
    int n;
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    Node** adjList = createGraph(n);
    generateRandomGraph(adjList, n);

    double start, end;

    start = omp_get_wtime();
    dfsSequential(adjList, n, 0);
    end = omp_get_wtime();
    double time_seq = end - start;
    printf("Sequential DFS Time: %.6f s\n", time_seq);

    start = omp_get_wtime();
    dfsParallel(adjList, n, 0);
    end = omp_get_wtime();
    double time_par = end - start;
    printf("Parallel DFS Time: %.6f s\n", time_par);
    printf("DFS Speedup: %.2fx\n", time_seq / time_par);

    for (int i = 0; i < n; i++) {
        Node* curr = adjList[i];
        while (curr) {
            Node* next = curr->next;
            free(curr);
            curr = next;
        }
    }
    free(adjList);

    return 0;
}

