#include "config.hpp"
#include "control_allocation.hpp"
#include "trim.hpp"
#include "lqr.hpp"
#include "pid.hpp"
#include "dynamics.hpp"
#include "integrator.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <cmath>
#include <cassert>

// Simple test framework
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "TEST: " << #name << " ... "; \
    try { test_##name(); tests_passed++; std::cout << "PASSED\n"; } \
    catch (const std::exception& e) { tests_failed++; std::cout << "FAILED: " << e.what() << "\n"; }

#define ASSERT_TRUE(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond)

#define ASSERT_NEAR(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        throw std::runtime_error( \
            "ASSERT_NEAR failed: " #a " = " + std::to_string(a) + \
            ", " #b " = " + std::to_string(b) + \
            ", diff = " + std::to_string(std::abs((a)-(b)))); \
    }

// Load default config for tests
static Config getTestConfig() {
    return loadConfig("config/tricopter_default.yaml");
}

// ===== Test: B_alloc dimensions and values =====
void test_balloc_dimensions() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    int N = static_cast<int>(cfg.rotors.size());
    ASSERT_TRUE(alloc.B_alloc.rows() == 6);
    ASSERT_TRUE(alloc.B_alloc.cols() == N);
    ASSERT_TRUE(alloc.B_force.rows() == 3);
    ASSERT_TRUE(alloc.B_force.cols() == N);
    ASSERT_TRUE(alloc.B_tau.rows() == 3);
    ASSERT_TRUE(alloc.B_tau.cols() == N);
}

void test_balloc_force_values() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    // For default config: all thrust axes = [0,0,1], k_T = 1.5e-5
    // Force column i = k_T * [0,0,1] = [0, 0, 1.5e-5]
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(alloc.B_force(0, i), 0.0, 1e-12);
        ASSERT_NEAR(alloc.B_force(1, i), 0.0, 1e-12);
        ASSERT_NEAR(alloc.B_force(2, i), 1.5e-5, 1e-12);
    }
}

void test_balloc_torque_values() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    // Rotor 0 (front): pos=[0.25,0,0], force=[0,0,k_T], spin_dir=+1, k_Q=2.5e-7
    // torque = cross([0.25,0,0], [0,0,1.5e-5]) + 1*2.5e-7*[0,0,1]
    //        = [0*1.5e-5 - 0*0, 0*0 - 0.25*1.5e-5, 0.25*0 - 0*0] + [0,0,2.5e-7]
    //        = [0, -3.75e-6, 0] + [0, 0, 2.5e-7]
    //        = [0, -3.75e-6, 2.5e-7]
    ASSERT_NEAR(alloc.B_tau(0, 0), 0.0, 1e-12);
    ASSERT_NEAR(alloc.B_tau(1, 0), -0.25 * 1.5e-5, 1e-12);
    ASSERT_NEAR(alloc.B_tau(2, 0), 2.5e-7, 1e-12);

    // Rotor 1 (rear-left): pos=[-0.15,0.20,0], spin_dir=-1
    // torque = cross([-0.15,0.20,0], [0,0,1.5e-5]) + (-1)*2.5e-7*[0,0,1]
    //        = [0.20*1.5e-5, -(-0.15)*1.5e-5, 0] + [0,0,-2.5e-7]
    //        = [3.0e-6, 2.25e-6, -2.5e-7]
    ASSERT_NEAR(alloc.B_tau(0, 1), 0.20 * 1.5e-5, 1e-12);
    ASSERT_NEAR(alloc.B_tau(1, 1), 0.15 * 1.5e-5, 1e-12);
    ASSERT_NEAR(alloc.B_tau(2, 1), -2.5e-7, 1e-12);
}

// ===== Test: Trim solver =====
void test_trim_residual_torque() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    TrimSolver trim;
    ASSERT_TRUE(trim.solve(cfg, alloc));

    // Residual torque is non-zero for this servoless tricopter geometry
    // (3 actuators, 4 constraints). The KKT solver minimises ||B_tau*u||.
    Eigen::Vector3d residual = alloc.B_tau * trim.u_hover;
    std::cout << "    Trim residual torque norm: " << residual.norm() << "\n";
    ASSERT_TRUE(std::isfinite(residual.norm()));
    ASSERT_TRUE(residual.norm() < 1.0);
}

void test_trim_thrust_balance() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    TrimSolver trim;
    ASSERT_TRUE(trim.solve(cfg, alloc));

    // Total thrust should equal m*g
    double mg = cfg.mass * 9.81;
    ASSERT_NEAR(trim.total_thrust_hover, mg, 0.01);
}

// ===== Test: CARE solver =====
void test_care_p_symmetric_positive_definite() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    AttitudeLQR lqr;
    ASSERT_TRUE(lqr.build(cfg, alloc));

    // P should be symmetric
    double sym_err = (lqr.P - lqr.P.transpose()).norm();
    ASSERT_TRUE(sym_err < 1e-8);

    // P should be positive definite (all eigenvalues > 0)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(lqr.P);
    for (int i = 0; i < 6; ++i) {
        ASSERT_TRUE(es.eigenvalues()(i) > 0);
    }
}

void test_care_closed_loop_stable() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    AttitudeLQR lqr;
    ASSERT_TRUE(lqr.build(cfg, alloc));

    // All closed-loop eigenvalues must have negative real part
    Eigen::Matrix<double, 6, 6> A_cl = lqr.A_att - lqr.B_att * lqr.K;
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> es(A_cl);
    for (int i = 0; i < 6; ++i) {
        ASSERT_TRUE(es.eigenvalues()(i).real() < 0);
    }
}

// ===== Test: Quaternion normalisation through RK4 =====
void test_quaternion_normalisation() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    TrimSolver trim;
    trim.solve(cfg, alloc);

    Eigen::VectorXd x = dynamics::buildInitialState(cfg);

    // Run 100 RK4 steps with hover trim
    integrator::DerivFunc f = dynamics::computeDerivative;
    for (int i = 0; i < 100; ++i) {
        x = integrator::rk4Step(f, x, trim.u_hover, cfg, cfg.sim.dt);
    }

    // Quaternion should remain normalised
    double qnorm = std::sqrt(x(6)*x(6) + x(7)*x(7) + x(8)*x(8) + x(9)*x(9));
    ASSERT_NEAR(qnorm, 1.0, 1e-10);
}

// ===== Test: PID step response =====
void test_pid_step_response() {
    AltitudePIDConfig pid_cfg;
    pid_cfg.Kp = 5.0;
    pid_cfg.Ki = 1.0;
    pid_cfg.Kd = 3.0;
    pid_cfg.integral_limit = 5.0;
    pid_cfg.output_limit = 10.0;

    PIDController pid;
    pid.init(pid_cfg);

    // Simulate a simple first-order system: dz/dt = u/m, with m=1.5
    double z = 0.0;
    double v = 0.0;
    double z_des = 1.0;
    double dt = 0.001;
    double m = 1.5;

    for (int i = 0; i < 10000; ++i) {
        double error = z_des - z;
        double u = pid.update(error, z, dt);
        // Simple double integrator: a = u/m
        double a = u / m;
        v += a * dt;
        z += v * dt;
    }

    // After 10s, should converge near setpoint
    ASSERT_NEAR(z, z_des, 0.1);
}

int main() {
    std::cout << "\n=== Running Unit Tests ===\n\n";

    TEST(balloc_dimensions);
    TEST(balloc_force_values);
    TEST(balloc_torque_values);
    TEST(trim_residual_torque);
    TEST(trim_thrust_balance);
    TEST(care_p_symmetric_positive_definite);
    TEST(care_closed_loop_stable);
    TEST(quaternion_normalisation);
    TEST(pid_step_response);

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "====================\n";

    return tests_failed > 0 ? 1 : 0;
}
