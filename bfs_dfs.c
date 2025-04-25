#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>

#define MAX_VERTICES 10000

// Graph using adjacency list
typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node* adjList[MAX_VERTICES];
bool visited[MAX_VERTICES];

// Add edge (undirected)
void addEdge(int u, int v) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = adjList[u];
    adjList[u] = newNode;

    newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = u;
    newNode->next = adjList[v];
    adjList[v] = newNode;
}

// ---------------------- BFS --------------------------

void bfsSequential(int start, int n) {
    bool visitedSeq[MAX_VERTICES] = {false};
    int queue[MAX_VERTICES];
    int front = 0, rear = 0;

    visitedSeq[start] = true;
    queue[rear++] = start;

    while (front < rear) {
        int curr = queue[front++];

        Node* temp = adjList[curr];
        while (temp != NULL) {
            int v = temp->vertex;
            if (!visitedSeq[v]) {
                visitedSeq[v] = true;
                queue[rear++] = v;
            }
            temp = temp->next;
        }
    }
}

void bfsParallel(int start, int n) {
    bool visitedPar[MAX_VERTICES] = {false};
    int queue[MAX_VERTICES];
    int front = 0, rear = 0;

    visitedPar[start] = true;
    queue[rear++] = start;

    while (front < rear) {
        int currentLevelSize = rear - front;

        #pragma omp parallel for shared(queue, visitedPar)
        for (int i = 0; i < currentLevelSize; i++) {
            int curr = queue[front + i];
            Node* temp = adjList[curr];

            while (temp != NULL) {
                int v = temp->vertex;

                if (!visitedPar[v]) {
                    #pragma omp critical
                    {
                        if (!visitedPar[v]) {
                            visitedPar[v] = true;
                            queue[rear++] = v;
                        }
                    }
                }
                temp = temp->next;
            }
        }
        front += currentLevelSize;
    }
}

// ---------------------- DFS --------------------------

void dfsSequentialUtil(int v, bool* visited) {
    visited[v] = true;
    Node* temp = adjList[v];

    while (temp != NULL) {
        if (!visited[temp->vertex]) {
            dfsSequentialUtil(temp->vertex, visited);
        }
        temp = temp->next;
    }
}

void dfsSequential(int start, int n) {
    bool visitedSeq[MAX_VERTICES] = {false};
    dfsSequentialUtil(start, visitedSeq);
}

void dfsParallelUtil(int v, bool* visited) {
    visited[v] = true;
    Node* temp = adjList[v];

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            while (temp != NULL) {
                int neighbor = temp->vertex;

                #pragma omp task firstprivate(neighbor)
                {
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                dfsParallelUtil(neighbor, visited);
                            }
                        }
                    }
                }
                temp = temp->next;
            }
        }
    }
}

void dfsParallel(int start, int n) {
    bool visitedPar[MAX_VERTICES] = {false};
    dfsParallelUtil(start, visitedPar);
}

// ---------------------- Driver --------------------------

int main() {
    int n = 10000;  // Number of vertices
    int edges = 20000;

    // Creating a random connected undirected graph
    for (int i = 0; i < n; i++) adjList[i] = NULL;
    for (int i = 1; i < n; i++) addEdge(i, i - 1); // Make sure it's connected
    for (int i = 0; i < edges; i++) {
        int u = rand() % n;
        int v = rand() % n;
        if (u != v) addEdge(u, v);
    }

    double startTime, endTime;

    // Sequential BFS
    startTime = omp_get_wtime();
    bfsSequential(0, n);
    endTime = omp_get_wtime();
    double bfs_seq_time = endTime - startTime;
    printf("Sequential BFS Time: %.4f seconds\n", bfs_seq_time);

    // Parallel BFS
    startTime = omp_get_wtime();
    bfsParallel(0, n);
    endTime = omp_get_wtime();
    double bfs_par_time = endTime - startTime;
    printf("Parallel BFS Time:   %.4f seconds\n", bfs_par_time);
    printf("BFS Speedup: %.2fx\n", bfs_seq_time / bfs_par_time);

    // Sequential DFS
    startTime = omp_get_wtime();
    dfsSequential(0, n);
    endTime = omp_get_wtime();
    double dfs_seq_time = endTime - startTime;
    printf("Sequential DFS Time: %.4f seconds\n", dfs_seq_time);

    // Parallel DFS
    startTime = omp_get_wtime();
    dfsParallel(0, n);
    endTime = omp_get_wtime();
    double dfs_par_time = endTime - startTime;
    printf("Parallel DFS Time:   %.4f seconds\n", dfs_par_time);
    printf("DFS Speedup: %.2fx\n", dfs_seq_time / dfs_par_time);

    return 0;
}

