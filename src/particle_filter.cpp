/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(int N, double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */

  // Set the number of particles
  num_particles = N;

  // Initialize a random engine
  std::default_random_engine gen;

  // Define normal distributions for x, y and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Creates N particles
  for (int i = 0; i < num_particles; ++i) {
    // Create a new particle
    Particle p;
    p.id = i;

    // Sample from the normal distributions for x, y and theta
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    // Initialize with equal weight of 1
    p.weight = 1.0;

    // Append this particle to the vector of particles
    particles.push_back(p);
  }

  // Set filter initialization true
  is_initialized = true;
  Print("Init", std::to_string(particles.size()) + " particles initialized");
}

// Add measurements to each particle and add random Gaussian noise to predict particle new state
// position/orientation
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Initialize a random engine
  std::default_random_engine gen;

  // Define normal distributions for x, y and theta random noise
  std::normal_distribution<double> random_gauss_x(0.0, std_pos[0]);
  std::normal_distribution<double> random_gauss_y(0.0, std_pos[1]);
  std::normal_distribution<double> random_gauss_theta(0.0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    // Calculate predicted positions/orientation and add random gaussian noise
    float x_pred =
        particles[i].x +
        velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) +
        random_gauss_x(gen);
    float y_pred =
        particles[i].y +
        velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) +
        random_gauss_y(gen);
    float theta_pred = particles[i].theta + yaw_rate * delta_t + random_gauss_theta(gen);

    // Constrain theta_pred between 0 and 2*PI
    theta_pred = fmod(theta_pred, 2.0 * M_PI);
    if (theta_pred < 0.0) theta_pred += 2.0 * M_PI;

    // Update particle with the predicted x, y and theta values
    particles[i].x = x_pred;
    particles[i].y = y_pred;
    particles[i].theta = theta_pred;
  }
  // Print("Predict",std::to_string(particles[0].id));
}

//
// Find and associate the nearst predicted map landmark to observations using the Nearst Neighbor data
// association technic
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (auto& observation : observations) {
    // Initialize euclidian distance error with max float value
    float best_error = std::numeric_limits<float>::infinity();

    // Iterate over each predicted map landmark
    for (auto& predicted_landmark : predicted) {
      // Calculate the euclidian distance
      float obs_error = dist(observation.x, observation.y, predicted_landmark.x, predicted_landmark.y);

      if (obs_error < best_error) {
        best_error = obs_error;

        // Update the observation landmark id with the current best predicted landmark
        observation.id = predicted_landmark.id;
      }
    }
  }
}

// Update the weights of each particle using a Multivariate-Gaussian distribution
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations, const Map& map_landmarks) {
  weights.clear();
  for (int i = 0; i < num_particles; ++i) {
    // Restart the particle weight
    particles[i].weight = 1;

    // Calculate the landmarks that are in the sensor range of the current predicted particle
    vector<LandmarkObs> predicted_landmarks{};
    for (auto map_landmark : map_landmarks.landmark_list) {
      float landmark_dist = dist(particles[i].x, particles[i].y, map_landmark.x_f, map_landmark.y_f);
      if (landmark_dist <= sensor_range) {
        LandmarkObs predicted_landmark;
        predicted_landmark.id = map_landmark.id_i;
        predicted_landmark.x = map_landmark.x_f;
        predicted_landmark.y = map_landmark.y_f;
        predicted_landmarks.push_back(predicted_landmark);
      }
    }

    // If there isn't any map landmark near the current particle, remove it from the particles vector
    if (predicted_landmarks.size() == 0) {
      particles.erase(particles.begin() + i);
      continue;
    }

    // Transform each observation to map coordinates
    vector<LandmarkObs> observations_map{};
    for (auto& observation : observations) {
      // transform current observation to map coordinates
      double observation_map_x = particles[i].x + (cos(particles[i].theta) * observation.x) -
                                 (sin(particles[i].theta) * observation.y);
      double observation_map_y = particles[i].y + (sin(particles[i].theta) * observation.x) +
                                 (cos(particles[i].theta) * observation.y);
      observations_map.push_back(LandmarkObs{observation.id, observation_map_x, observation_map_y});
    }

    // Associate the nearst landmark on the map to each transformed observation
    ParticleFilter::dataAssociation(predicted_landmarks, observations_map);

    std::vector<int> association_landmark_id{};
    std::vector<double> association_sense_x{};
    std::vector<double> association_sense_y{};

    for (auto& observation : observations_map) {
      association_landmark_id.push_back(observation.id);
      association_sense_x.push_back(observation.x);
      association_sense_y.push_back(observation.y);

      // Find the correspondent map landmark to the current observation
      std::vector<LandmarkObs>::iterator it = std::find_if(
          predicted_landmarks.begin(), predicted_landmarks.end(), landmarkFinder(observation.id));

      // Calculate the particle weights using the Multivariate-Gaussian probability density function
      particles[i].weight *=
          multiv_prob(std_landmark[0], std_landmark[1], observation.x, observation.y, it->x, it->y);
    }

    weights.push_back(particles[i].weight);
    // Set particle association vectors
    SetAssociations(particles[i], association_landmark_id, association_sense_x, association_sense_y);
  }

  // Compute normalization factor
  double weights_sum = 0.0;
  for (int i = 0; i < weights.size(); ++i) {
    weights_sum += weights[i];
  }

  // Normalize weights and update the weights vector in same order as particles vector
  for (int i = 0; i < weights.size(); ++i) {
    weights[i] /= weights_sum;
  }
}

// Resample particles with replacement with probability proportional to their weight.
void ParticleFilter::resample() {
  // Initialize the discrete distribution with the weights vector
  std::default_random_engine gen;
  std::discrete_distribution<> distribution(weights.begin(), weights.end());

  std::vector<Particle> resampled_particles{};
  // Resample particles using the discrete distribution
  for (int i = 0; i < num_particles; i++) {
    int particle_index = distribution(gen);
    resampled_particles.push_back(particles[particle_index]);
  }

  // Update particles vector
  particles = resampled_particles;
}

// Set a particles list of associations
void ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations,
                                     const vector<double>& sense_x, const vector<double>& sense_y) {
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}