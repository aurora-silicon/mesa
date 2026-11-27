/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

struct hk_device;
struct hk_queue;

/* Sets the kicker up for a G17C device whose hk_compute_kick_ms is not 0.
 * Called once the device's queues exist; never fails device creation. */
void hk_compute_kick_init(struct hk_device *dev, unsigned period_ms);

/* Stops and joins the kicker thread, waits for the last kicks and destroys
 * the kick queues and objects. Called before anything else of the device is
 * destroyed. Notes that arrive afterwards change nothing. */
void hk_compute_kick_stop(struct hk_device *dev);

/* Frees the rest. Called once the application's queues are finished, so
 * that no submit thread can be in hk_compute_kick_note. */
void hk_compute_kick_finish(struct hk_device *dev);

/* Called from the queue's submission callback after compute work was
 * submitted to a watched queue, with its timeline value already advanced. */
void hk_compute_kick_note(struct hk_device *dev, struct hk_queue *queue);

/* Guard a single-queue packed global-atomic submission against concurrent
 * kick-queue activation. Returns true when the queue is watched, no kick
 * queue has been activated, and the caller has registered an in-flight
 * packed-atomics submission (which must be paired with
 * hk_compute_kick_pack_atomics_end()). */
bool hk_compute_kick_pack_atomics_begin(struct hk_device *dev,
                                        struct hk_queue *queue);
void hk_compute_kick_pack_atomics_end(struct hk_device *dev,
                                      struct hk_queue *queue,
                                      uint64_t timeline_point);
