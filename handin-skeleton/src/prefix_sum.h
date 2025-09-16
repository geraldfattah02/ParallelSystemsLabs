#pragma once

#include "helpers.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include "prefix_sum.h"
#include "barrier.h"
#include <cstdio>


void* compute_prefix_sum(void* a);
