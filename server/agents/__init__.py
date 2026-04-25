"""
ClusterEnv agent implementations.

  CoolingHeuristic          — rule-based proportional fan/setpoint controller
  OversightMonitor          — rule-based gaming detection (Fleet AI sub-theme)
  priority_weighted_threshold — baseline admission scheduler (for calibration)
  accept_all, reject_all      — extreme baselines for bounds checking
  CooperativeTeam     — honest team_a job generator (8 archetypes, carbon-aware)
  StrategicTeam       — gaming team_b job generator (always inflates priority/deadline)
"""
from .cooling_heuristic import CoolingHeuristic
from .oversight_monitor import OversightMonitor
from .baseline_scheduler import priority_weighted_threshold, accept_all, reject_all
from .scripted_teams import CooperativeTeam, StrategicTeam
