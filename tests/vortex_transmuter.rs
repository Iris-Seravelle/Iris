#![cfg(feature = "vortex")]

use iris::vortex::{VortexInstruction, VortexSuspend, VortexTransmuter};

#[test]
fn vortex_transmuter_budget_allocation() {
    let mut t = VortexTransmuter::new(100);
    assert!(t.inject_reduction_checks(20));
    assert_eq!(t.instruction_budget, 80);
    assert!(t.inject_reduction_checks(80));
    assert_eq!(t.instruction_budget, 0);
    assert!(!t.inject_reduction_checks(1));
    assert!(!t.enabled);
}

#[test]
fn vortex_transmuter_transmute_injects_entry_and_loop_checks() {
    let transmuter = VortexTransmuter::new(10);
    let code = vec![
        VortexInstruction::LoadFast(0),
        VortexInstruction::JumpBackward(0),
        VortexInstruction::ReturnValue,
    ];
    let transmuted = transmuter.transmute(&code);

    assert_eq!(transmuted[0], VortexInstruction::IrisReductionCheck);
    assert!(transmuted.contains(&VortexInstruction::IrisReductionCheck));
    assert!(transmuted.len() > code.len());
}

#[test]
fn vortex_transmuter_execute_suspends_on_budget_exhaustion() {
    let mut transmuter = VortexTransmuter::new(1);
    let code = vec![
        VortexInstruction::IrisReductionCheck,
        VortexInstruction::LoadFast(0),
        VortexInstruction::ReturnValue,
    ];

    let result = transmuter.execute(&code);

    assert_eq!(result, Err(VortexSuspend));
}

#[test]
fn vortex_transmuter_execute_completes() {
    let mut transmuter = VortexTransmuter::new(10);
    let code = vec![
        VortexInstruction::IrisReductionCheck,
        VortexInstruction::LoadFast(0),
        VortexInstruction::ReturnValue,
    ];

    assert_eq!(transmuter.execute(&code), Ok(()));
}

#[test]
fn vortex_quiescence_points_found() {
    let transmuter = VortexTransmuter::new(10);
    let code = vec![
        VortexInstruction::LoadFast(0),
        VortexInstruction::StoreFast(0),
        VortexInstruction::ReturnValue,
    ];
    let points = transmuter.quiescence_points(&code);

    assert!(points.contains(&1));
    assert!(points.contains(&2));
}
