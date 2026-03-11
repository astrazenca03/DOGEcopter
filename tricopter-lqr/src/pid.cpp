#include "pid.hpp"
#include <algorithm>
#include <cmath>

void PIDController::init(const AltitudePIDConfig& cfg) {
    Kp_ = cfg.Kp;
    Ki_ = cfg.Ki;
    Kd_ = cfg.Kd;
    integral_limit_ = cfg.integral_limit;
    output_limit_ = cfg.output_limit;
    reset();
}

void PIDController::reset() {
    integral_ = 0.0;
    prev_measurement_ = 0.0;
    first_update_ = true;
}

double PIDController::update(double error, double measurement, double dt) {
    if (dt <= 0.0) return 0.0;

    // Proportional
    double P_term = Kp_ * error;

    // Integral with anti-windup (clamp)
    integral_ += Ki_ * error * dt;
    integral_ = std::clamp(integral_, -integral_limit_, integral_limit_);

    // Derivative on measurement (not error) to avoid derivative kick
    double D_term = 0.0;
    if (!first_update_) {
        double d_measurement = (measurement - prev_measurement_) / dt;
        D_term = -Kd_ * d_measurement;  // negative because derivative on measurement
    }
    prev_measurement_ = measurement;
    first_update_ = false;

    // Sum and clamp output
    double output = P_term + integral_ + D_term;
    output = std::clamp(output, -output_limit_, output_limit_);
    return output;
}
