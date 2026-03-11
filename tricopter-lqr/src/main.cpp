#include "config.hpp"
#include "control_allocation.hpp"
#include "trim.hpp"
#include "lqr.hpp"
#include "simulation.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string config_path = "config/tricopter_default.yaml";
    if (argc > 1)
        config_path = argv[1];

    std::cout << "========================================\n";
    std::cout << "  Tricopter LQR Flight Controller\n";
    std::cout << "  Cascaded: Attitude LQR + Altitude PID\n";
    std::cout << "========================================\n\n";

    // 1. Load config
    Config cfg;
    try {
        cfg = loadConfig(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    // 2. Build control allocation
    ControlAllocation alloc;
    alloc.build(cfg);
    alloc.printDiagnostics();

    // 3. Solve trim
    TrimSolver trim;
    if (!trim.solve(cfg, alloc)) {
        std::cerr << "Trim solver failed or produced invalid results.\n";
        return 1;
    }

    // 4. Build attitude LQR
    AttitudeLQR lqr;
    if (!lqr.build(cfg, alloc)) {
        std::cerr << "LQR build failed — system is unstable!\n";
        return 1;
    }

    // 5. Run simulation
    Simulation sim;
    sim.run(cfg, alloc, trim, lqr, "sim_output.csv");

    return 0;
}
