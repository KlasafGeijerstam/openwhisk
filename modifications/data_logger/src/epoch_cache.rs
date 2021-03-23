use std::fmt;

pub struct EpochCache {
    epoch_length: usize,
    current: u64,
    working: Vec<u64>,
}

impl Default for EpochCache {
    fn default() -> Self {
        EpochCache::new()
    }
}

impl fmt::Display for EpochCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::fmt::Debug for EpochCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EpochCache")
            .field("epoch_length", &self.epoch_length)
            .field("current", &self.current)
            .field("working_len", &self.working.len())
            .finish()
    }
}

impl EpochCache {
    pub fn new() -> Self {
        EpochCache {
            epoch_length: 10,
            current: 0,
            working: Vec::new(),
        }
    }

    pub fn add(&mut self, value: u64) {
        self.working.push(value);

        if self.current == 0 && self.working.len() < self.epoch_length {
            self.current = self.moving_average();
        } else if self.working.len() == self.epoch_length {
            self.current = self.moving_average();

            self.working.clear();
        }
    }

    pub fn current(&self) -> u64 {
        self.current
    }

    fn moving_average(&self) -> u64 {
        if self.working.is_empty() {
            return 0;
        }

        let alpha = 0.5;

        let avg = self
            .working
            .iter()
            .skip(1)
            .fold(self.working[0] as f64, |avg, &cur| {
                alpha * (cur as f64) + (1.0 - alpha) * avg
            });

        avg as u64
    }
}
