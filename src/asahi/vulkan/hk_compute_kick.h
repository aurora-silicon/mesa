/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

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
