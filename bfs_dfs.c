#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <time.h>

#define MAX_VERTICES 10000

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node* adjList[MAX_VERTICES];

void addEdge(int u, int v) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->vertex = v;
    node->next = adjList[u];
    adjList[u] = node;

    node = (Node*)malloc(sizeof(Node));
    node->vertex = u;
    node->next = adjList[v];
    adjList[v] = node;
}

bool edgeExists(int u, int v) {
    Node* temp = adjList[u];
    while (temp) {
        if (temp->vertex == v) return true;
        temp = temp->next;
    }
    return false;
}

void generateRandomGraph(int n, int edges) {
    srand(time(NULL));
    int count = 0;
    while (count < edges) {
        int u = rand() % n;
        int v = rand() % n;
        if (u != v && !edgeExists(u, v)) {
            addEdge(u, v);
            count++;
        }
    }
}

// -------------------- BFS --------------------
void bfsSequential(int start, int n) {
    bool visited[MAX_VERTICES] = {false};
    int queue[MAX_VERTICES], front = 0, rear = 0;

    visited[start] = true;
    queue[rear++] = start;

    while (front < rear) {
        int curr = queue[front++];
        Node* temp = adjList[curr];
        while (temp) {
            int v = temp->vertex;
            if (!visited[v]) {
                visited[v] = true;
                queue[rear++] = v;
            }
            temp = temp->next;
        }
    }
}

void bfsParallel(int start, int n) {
    bool visited[MAX_VERTICES] = {false};
    int queue[MAX_VERTICES], front = 0, rear = 0;

    visited[start] = true;
    queue[rear++] = start;

    while (front < rear) {
        int currentSize = rear - front;

        #pragma omp parallel for
        for (int i = 0; i < currentSize; i++) {
            int node = queue[front + i];
            Node* temp = adjList[node];

            while (temp) {
                int v = temp->vertex;
                #pragma omp critical
                {
                    if (!visited[v]) {
                        visited[v] = true;
                        queue[rear++] = v;
                    }
                }
                temp = temp->next;
            }
        }

        front += currentSize;
    }
}

// -------------------- DFS --------------------
void dfsSequentialUtil(int v, bool visited[]) {
    visited[v] = true;
    Node* temp = adjList[v];
    while (temp) {
        if (!visited[temp->vertex])
            dfsSequentialUtil(temp->vertex, visited);
        temp = temp->next;
    }
}

void dfsSequential(int start, int n) {
    bool visited[MAX_VERTICES] = {false};
    dfsSequentialUtil(start, visited);
}

void dfsParallelUtil(int v, bool visited[]) {
    bool proceed = false;

    #pragma omp critical
    {
        if (!visited[v]) {
            visited[v] = true;
            proceed = true;
        }
    }

    if (!proceed) return;

    Node* temp = adjList[v];

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (Node* p = temp; p != NULL; p = p->next) {
                int neighbor = p->vertex;
                #pragma omp task firstprivate(neighbor)
                dfsParallelUtil(neighbor, visited);
            }
        }
    }
}

void dfsParallel(int start, int n) {
    bool visited[MAX_VERTICES] = {false};
    dfsParallelUtil(start, visited);
}

// -------------------- MAIN --------------------
int main() {
    int n, edges;
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter number of edges: ");
    scanf("%d", &edges);

    for (int i = 0; i < n; i++) adjList[i] = NULL;

    printf("Generating random graph with %d vertices and %d edges...\n", n, edges);
    generateRandomGraph(n, edges);

    double t1, t2;

    t1 = omp_get_wtime();
    bfsSequential(0, n);
    t2 = omp_get_wtime();
    double bfs_seq = t2 - t1;
    printf("Sequential BFS Time: %.6f\n", bfs_seq);

    t1 = omp_get_wtime();
    bfsParallel(0, n);
    t2 = omp_get_wtime();
    double bfs_par = t2 - t1;
    printf("Parallel BFS Time:   %.6f\n", bfs_par);
    printf("BFS Speedup: %.2fx\n", bfs_seq / bfs_par);

    t1 = omp_get_wtime();
    dfsSequential(0, n);
    t2 = omp_get_wtime();
    double dfs_seq = t2 - t1;
    printf("Sequential DFS Time: %.6f\n", dfs_seq);

    t1 = omp_get_wtime();
    dfsParallel(0, n);
    t2 = omp_get_wtime();
    double dfs_par = t2 - t1;
    printf("Parallel DFS Time:   %.6f\n", dfs_par);
    printf("DFS Speedup: %.2fx\n", dfs_seq / dfs_par);

    return 0;
}

