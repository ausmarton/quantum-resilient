//! Experiment Scheduler
//!
//! Provides cron-like scheduling for automated experiment runs.

use chrono::{DateTime, Utc};
use cron::Schedule;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

#[derive(Error, Debug)]
pub enum SchedulerError {
    #[error("Invalid cron expression: {0}")]
    InvalidCronExpression(String),
    #[error("Schedule not found: {0}")]
    ScheduleNotFound(String),
    #[error("Schedule already exists: {0}")]
    ScheduleAlreadyExists(String),
}

/// A scheduled experiment definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScheduledExperiment {
    /// Unique name for this schedule
    pub name: String,
    /// Cron expression (e.g., "0 3 * * *" for 3 AM daily)
    pub cron_expr: String,
    /// Scenario YAML configuration
    pub scenario_yaml: String,
    /// Number of worker replicas
    pub replicas: u32,
    /// Whether the schedule is enabled
    pub enabled: bool,
    /// When this schedule was created
    pub created_at: DateTime<Utc>,
    /// Last time this schedule was triggered
    pub last_run: Option<DateTime<Utc>>,
    /// Next scheduled run time
    pub next_run: Option<DateTime<Utc>>,
    /// Last experiment ID created by this schedule
    pub last_experiment_id: Option<String>,
}

/// Request to create a new scheduled experiment
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateScheduleRequest {
    pub name: String,
    pub cron: String,
    pub scenario_config: String,
    pub replicas: u32,
}

/// The experiment scheduler
pub struct ExperimentScheduler {
    schedules: Arc<RwLock<HashMap<String, ScheduledExperiment>>>,
    trigger_tx: Option<mpsc::Sender<String>>,
}

impl ExperimentScheduler {
    /// Parse a cron expression, accepting either standard 6-field or 5-field (minute-level) forms.
    /// For 5-field inputs we prepend a leading seconds field so `cron` can parse it.
    fn parse_cron_expr(expr: &str) -> Result<(String, Schedule), SchedulerError> {
        let trimmed = expr.trim();

        match Schedule::from_str(trimmed) {
            Ok(schedule) => Ok((trimmed.to_string(), schedule)),
            Err(initial_err) => {
                // Allow 5-field expressions like "0 3 * * *" by assuming 0 seconds.
                let parts: Vec<_> = trimmed.split_whitespace().collect();
                if parts.len() == 5 {
                    let with_seconds = format!("0 {}", trimmed);
                    Schedule::from_str(&with_seconds)
                        .map(|schedule| (with_seconds, schedule))
                        .map_err(|_| SchedulerError::InvalidCronExpression(initial_err.to_string()))
                } else {
                    Err(SchedulerError::InvalidCronExpression(initial_err.to_string()))
                }
            }
        }
    }

    pub fn new() -> Self {
        Self {
            schedules: Arc::new(RwLock::new(HashMap::new())),
            trigger_tx: None,
        }
    }

    /// Set the trigger channel for firing experiments
    pub fn set_trigger_channel(&mut self, tx: mpsc::Sender<String>) {
        self.trigger_tx = Some(tx);
    }

    /// Add a new scheduled experiment
    pub fn add_schedule(&self, request: CreateScheduleRequest) -> Result<ScheduledExperiment, SchedulerError> {
        // Validate cron expression
        let (cron_expr, schedule) = Self::parse_cron_expr(&request.cron)?;

        // Calculate next run time
        let next_run = schedule.upcoming(Utc).next();

        let scheduled = ScheduledExperiment {
            name: request.name.clone(),
            cron_expr,
            scenario_yaml: request.scenario_config,
            replicas: request.replicas,
            enabled: true,
            created_at: Utc::now(),
            last_run: None,
            next_run,
            last_experiment_id: None,
        };

        {
            let mut schedules = self.schedules.write();
            if schedules.contains_key(&request.name) {
                return Err(SchedulerError::ScheduleAlreadyExists(request.name));
            }
            schedules.insert(request.name, scheduled.clone());
        }

        info!("Added schedule: {} (next run: {:?})", scheduled.name, scheduled.next_run);
        Ok(scheduled)
    }

    /// Remove a scheduled experiment
    pub fn remove_schedule(&self, name: &str) -> Result<(), SchedulerError> {
        let mut schedules = self.schedules.write();
        schedules
            .remove(name)
            .ok_or_else(|| SchedulerError::ScheduleNotFound(name.to_string()))?;
        info!("Removed schedule: {}", name);
        Ok(())
    }

    /// Get a schedule by name
    pub fn get_schedule(&self, name: &str) -> Option<ScheduledExperiment> {
        let schedules = self.schedules.read();
        schedules.get(name).cloned()
    }

    /// List all schedules
    pub fn list_schedules(&self) -> Vec<ScheduledExperiment> {
        let schedules = self.schedules.read();
        schedules.values().cloned().collect()
    }

    /// Enable or disable a schedule
    pub fn set_enabled(&self, name: &str, enabled: bool) -> Result<(), SchedulerError> {
        let mut schedules = self.schedules.write();
        let schedule = schedules
            .get_mut(name)
            .ok_or_else(|| SchedulerError::ScheduleNotFound(name.to_string()))?;
        schedule.enabled = enabled;
        
        // Update next run time if enabling
        if enabled {
            if let Ok(cron_schedule) = Schedule::from_str(&schedule.cron_expr) {
                schedule.next_run = cron_schedule.upcoming(Utc).next();
            }
        } else {
            schedule.next_run = None;
        }
        
        Ok(())
    }

    /// Check and trigger any due schedules
    /// Returns the names of schedules that were triggered
    pub fn check_and_trigger(&self) -> Vec<String> {
        let now = Utc::now();
        let mut triggered = Vec::new();

        let mut schedules = self.schedules.write();
        for (name, schedule) in schedules.iter_mut() {
            if !schedule.enabled {
                continue;
            }

            if let Some(next_run) = schedule.next_run {
                if now >= next_run {
                    info!("Triggering scheduled experiment: {}", name);
                    triggered.push(name.clone());
                    schedule.last_run = Some(now);

                    // Calculate next run time
                    if let Ok(cron_schedule) = Schedule::from_str(&schedule.cron_expr) {
                        schedule.next_run = cron_schedule.upcoming(Utc).next();
                    }
                }
            }
        }

        triggered
    }

    /// Update a schedule after an experiment is created
    pub fn update_last_experiment(&self, name: &str, experiment_id: &str) {
        let mut schedules = self.schedules.write();
        if let Some(schedule) = schedules.get_mut(name) {
            schedule.last_experiment_id = Some(experiment_id.to_string());
        }
    }

    /// Get schedules data for persistence
    pub fn get_schedules_data(&self) -> HashMap<String, ScheduledExperiment> {
        let schedules = self.schedules.read();
        schedules.clone()
    }

    /// Load schedules from persisted data
    pub fn load_schedules(&self, data: HashMap<String, ScheduledExperiment>) {
        let mut schedules = self.schedules.write();
        *schedules = data;
        
        // Update next run times for enabled schedules
        for schedule in schedules.values_mut() {
            if schedule.enabled {
                if let Ok(cron_schedule) = Schedule::from_str(&schedule.cron_expr) {
                    schedule.next_run = cron_schedule.upcoming(Utc).next();
                }
            }
        }
        
        info!("Loaded {} schedules", schedules.len());
    }
}

impl Default for ExperimentScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Background task to check schedules every minute
pub async fn scheduler_loop(
    scheduler: Arc<ExperimentScheduler>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
    trigger_tx: mpsc::Sender<(String, String, u32)>, // (schedule_name, scenario_yaml, replicas)
) {
    info!("Scheduler loop started");
    
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    
    loop {
        tokio::select! {
            _ = interval.tick() => {
                let triggered = scheduler.check_and_trigger();
                
                for name in triggered {
                    if let Some(schedule) = scheduler.get_schedule(&name) {
                        if let Err(e) = trigger_tx.send((
                            name.clone(),
                            schedule.scenario_yaml.clone(),
                            schedule.replicas,
                        )).await {
                            error!("Failed to send trigger for schedule {}: {}", name, e);
                        }
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                info!("Scheduler loop shutting down");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_schedule() {
        let scheduler = ExperimentScheduler::new();
        
        let request = CreateScheduleRequest {
            name: "test_schedule".to_string(),
            cron: "0 3 * * *".to_string(), // 3 AM daily
            scenario_config: "id: test\nworkload:\n  msgs_per_sec: 100".to_string(),
            replicas: 5,
        };
        
        let result = scheduler.add_schedule(request);
        assert!(result.is_ok());
        
        let schedule = result.unwrap();
        assert_eq!(schedule.name, "test_schedule");
        assert!(schedule.next_run.is_some());
    }

    #[test]
    fn test_invalid_cron() {
        let scheduler = ExperimentScheduler::new();
        
        let request = CreateScheduleRequest {
            name: "invalid".to_string(),
            cron: "not a cron expression".to_string(),
            scenario_config: "".to_string(),
            replicas: 1,
        };
        
        let result = scheduler.add_schedule(request);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_schedules() {
        let scheduler = ExperimentScheduler::new();
        
        let request = CreateScheduleRequest {
            name: "schedule1".to_string(),
            cron: "0 * * * *".to_string(), // Every hour
            scenario_config: "".to_string(),
            replicas: 1,
        };
        scheduler.add_schedule(request).unwrap();
        
        let schedules = scheduler.list_schedules();
        assert_eq!(schedules.len(), 1);
    }
}

