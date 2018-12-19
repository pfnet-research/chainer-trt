/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

#include "chainer_trt/plugin.hpp"

#include "argmax.hpp"
#include "broadcast_to.hpp"
#include "constant_elementwise.hpp"
#include "directed_pooling.hpp"
#include "leaky_relu.hpp"
#include "resize.hpp"
#include "resize_argmax.hpp"
#include "shift.hpp"
#include "slice.hpp"
#include "sum.hpp"
#include "where.hpp"
