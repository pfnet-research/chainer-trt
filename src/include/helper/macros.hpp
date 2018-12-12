/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#define CHECK_CUDA(result, message)     \
    {                                   \
        if(result != cudaSuccess)       \
            throw std::string(message); \
    }