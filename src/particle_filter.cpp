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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  particles.clear();  // Making sure particles vector is empty before populating
  weights.clear();    // This will be filled in UpdateWeights
  
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> x_gen{x, std[0]};
  std::normal_distribution<double> y_gen{y, std[1]};
  std::normal_distribution<double> theta_gen{theta, std[2]};
  
  for (int i = 0; i < num_particles; ++i)
  {
    particles.push_back( Particle{i, x_gen(gen), y_gen(gen), theta_gen(gen), 1.0} );
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> x_gen{0.0, std_pos[0]};
  std::normal_distribution<double> y_gen{0.0, std_pos[1]};
  std::normal_distribution<double> theta_gen{0.0, std_pos[2]};
  
  // epsilon for doubles is order of magnitude E-16
  if (std::fabs(yaw_rate) > std::numeric_limits<double>::epsilon()) {
    double vel_ratio = velocity / yaw_rate;
    double dtheta = yaw_rate * delta_t;
  
    for (Particle& p: particles) {
      p.x += vel_ratio * (std::sin(p.theta + dtheta) - std::sin(p.theta)) + x_gen(gen);
      p.y += vel_ratio * (std::cos(p.theta) - std::cos(p.theta + dtheta)) + y_gen(gen);
      p.theta += dtheta + theta_gen(gen);
    }
  }
  else {
    double ds = velocity * delta_t;
    for (Particle& p: particles) {
      p.x += ds * std::cos(p.theta) + x_gen(gen);
      p.y += ds * std::sin(p.theta) + y_gen(gen);
      p.theta += theta_gen(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (LandmarkObs& obs: observations) {
    double min_dist = std::numeric_limits<double>::infinity();
    double dist_landmark = 0.0;
    int min_id = 0;
    
    for (LandmarkObs& pred: predicted) {
      dist_landmark = dist(obs.x, obs.y, pred.x, pred.y);
      if (dist_landmark < min_dist) {
         min_dist = dist_landmark;
         min_id   = pred.id;
      }
    }
    obs.id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  weights.clear();
  // For each particle
  for (Particle& p: particles) {
    // Get all landmarks within sensor range in MAP coords
    vector<LandmarkObs> predicted;
    for (const Map::single_landmark_s& L: map_landmarks.landmark_list) {
      if (dist(p.x, p.y, L.x_f, L.y_f) <= sensor_range) 
        predicted.push_back(LandmarkObs{L.id_i, L.x_f, L.y_f});
    }
    
    // Transform observations in MAP coords and associate using nearest neighbour metric
    double cos_theta = std::cos(p.theta), sin_theta = std::sin(p.theta);
    
    vector<LandmarkObs> particle_obs;
    std::transform(observations.begin(),
                   observations.end(),
                   std::back_inserter(particle_obs),
                   [&p, &cos_theta, &sin_theta](const LandmarkObs& L_robot)
                   {
                     return LandmarkObs{L_robot.id,
                                        cos_theta * L_robot.x - sin_theta * L_robot.y + p.x,
                                        sin_theta * L_robot.x + cos_theta * L_robot.y + p.y};
                   });
    
    // particle_obs ID shows nearest predicted landmark
    dataAssociation(predicted, particle_obs);
    
    // Update weight
    double gauss_norm = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]),
           sx2 = 2.0 * std_landmark[0] * std_landmark[0], sy2 = 2.0 * std_landmark[1] * std_landmark[1];
    
    p.weight = 1.0;
    for (const LandmarkObs& obs: particle_obs) {
      
      for (const LandmarkObs& pred: predicted) {
        if (pred.id == obs.id) { // For the associated predicted measurement compute the probability and update the weight
          double dx2 = (pred.x - obs.x)*(pred.x - obs.x)/sx2, dy2 = (pred.y - obs.y)*(pred.y - obs.y)/sy2;
          double update = gauss_norm * std::exp( - (dx2 + dy2) );
          
          if (update < std::numeric_limits<double>::epsilon()) p.weight *= std::numeric_limits<double>::epsilon();
          else p.weight *= update;
          
          break;
        }
      }
      
    }
    // Fill in the weights vector
    weights.push_back(p.weight);
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Discrete distribution samples based on the normalized weights
  // so we don't need to normalize manually
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> sample(weights.begin(), weights.end());
  
  vector<Particle> resampled;
  
  for (unsigned int i = 0; i < particles.size(); ++i) {
    int ind = sample(gen);
    resampled.push_back(particles[ind]);
  }
  
  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
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
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}