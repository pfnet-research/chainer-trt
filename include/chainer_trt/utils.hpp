/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

void show_device_info(int gpu_id = -1);

void hwc2chw(float* dst, const float* src, int w, int h);
