#pragma once

#include "config.hpp"

/// Standard PID controller with anti-windup and derivative-on-measurement.
///
/// Used as the outer-loop altitude controller.
///   Input:  altitude error = z_desired - z_current
///   Output: thrust adjustment added to m*g
class PIDController {
public:
    PIDController() = default;

    /// Initialise from config.
    void init(const AltitudePIDConfig& cfg);

    /// Reset integrator and stored state.
    void reset();

    /// Update PID. Returns control output.
    ///   error     = setpoint - measurement
    ///   measurement = current altitude (for derivative-on-measurement)
    ///   dt        = timestep
    double update(double error, double measurement, double dt);

private:
    double Kp_ = 0;
    double Ki_ = 0;
    double Kd_ = 0;
    double integral_limit_ = 0;
    double output_limit_ = 0;
    double integral_ = 0;
    double prev_measurement_ = 0;
    bool first_update_ = true;
};
