#include <gtest/gtest.h>
#

__global__ void precision_test(float* dest)
{
    dest[0] = .3f + .5f;
}

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {

    
    float* host = (float *)malloc(sizeof(float));
    float* device = nullptr;
    cudaMalloc(&device, sizeof(float));
    cudaMemcpy(device, host, sizeof(float), cudaMemcpyHostToDevice);

    precision_test<<<1, 1>>>(device);

    cudaMemcpy(host, device, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", host[0]);

    free(host);
    cudaFree(device);

    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}