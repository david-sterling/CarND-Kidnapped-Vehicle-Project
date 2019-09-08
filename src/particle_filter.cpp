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
// Trying to avoid nasty things including asserts
#include <assert.h>
//Random number generator
static std::default_random_engine gen;
#define YAW_MIN 0.001
#define NUMPARTICLES 1000
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

	//Check for inconsistencies before starting
	assert(std[0] != 0 && std[1] != 0 && std[2] != 0);
	assert(is_initialized == false);

	num_particles = NUMPARTICLES; //Set the number of particles
  // Going crazy is not advised since this C++ program is not multicore enabled

  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std[0]);

  // Create normal distributions for y and theta
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Start instantiating particle structs. Avoid rellocating memory to gain speed.
  particles.resize(num_particles);
  for (int i = 0; i < num_particles; i++) {
	  //struct Particle {int id;double x;double y;double theta;double weight; std::vector<int> associations;std::vector<double> sense_x;std::vector<double> sense_y;
	  
	  Particle newParticle = {i,dist_x(gen),dist_y(gen),dist_theta(gen),1.0};
	  particles[i] = newParticle;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */


	// Basic consistency checks
	assert(std_pos[0] != 0 && std_pos[1] != 0 && std_pos[2] != 0);
	assert(is_initialized == true);

	// Now we update the particles

	for (int i = 0; i < num_particles; i++) {
		//Two cases here, with yaw rate or without yaw rate (equations change)
		if (fabs(yaw_rate) < YAW_MIN) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
	}
	// Generate gaussian noise distributions
	std::normal_distribution<double> gnoise_x(0, std_pos[0]);
	std::normal_distribution<double> gnoise_y(0, std_pos[1]);
	std::normal_distribution<double> gnoise_theta(0, std_pos[2]);

	// Apply gaussian noise
	for (int i = 0; i < num_particles; i++) {
		particles[i].x +=  gnoise_x(gen);
		particles[i].y +=  gnoise_y(gen);
		particles[i].theta +=  gnoise_theta(gen);
	}

	
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

	// Go through all observations vector
	for (unsigned int i = 0; i < observations.size(); i++) {
		//Initialize variable with maximum value
		double min_dist = std::numeric_limits<double>::max();
		//Initialize with negative to show ID is wrong
		int map_id = -1;
		for (unsigned int j = 0; j < predicted.size(); j++) {
			//Compute the euclidean distance from observation to predicted
			double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			//Now check the minimum and update if necessary
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				map_id = predicted[j].id;
			}
		}
		//Assign the nearest landmark
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const vector<LandmarkObs>& observations,
	const Map& map_landmarks) {
	/**
	 *  Update the weights of each particle using a mult-variate Gaussian
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


	 /*	1 Transformation
		 2 Association
		 3 Update Weights
		 */

	//Some checks
	assert(is_initialized == true && sensor_range != 0.00);
	for (int i = 0; i < num_particles; i++) {
		double theta = particles[i].theta;
		// Find landmarks in particle's range, sensor range is given in function parameters
		vector<LandmarkObs> nearlandmarks;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			// Check if point is within sensor range, compute euclidean distance
			if(dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range){
				nearlandmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f });
			}
		}

		//Transform from car axis to map axis
		vector<LandmarkObs> trfObservations;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double xx = cos(theta) * observations[j].x - sin(theta) * observations[j].y + particles[i].x;
			double yy = sin(theta) * observations[j].x + cos(theta) * observations[j].y + particles[i].y;
			trfObservations.push_back(LandmarkObs{ observations[j].id, xx, yy });
		}

		// Observation association to landmark using the function coded before
		dataAssociation(nearlandmarks, trfObservations);

		//Now the weigths computing part
		particles[i].weight = 1.0;
		for (unsigned int j = 0; j < trfObservations.size(); j++) {
			LandmarkObs lmfound;
			unsigned int k = 0;
			bool found = false;
			//Break if found, save a few CPU cycles
			while (!found && k < nearlandmarks.size()) {
				if (nearlandmarks[k].id == trfObservations[j].id) {
					found = true;
					lmfound = nearlandmarks[k];
				}
				k++;
			}

			//I use the lesson 6 code multiv_prob
			double dX = trfObservations[j].x - lmfound.x;
			double dY = trfObservations[j].y - lmfound.y;

			double weight = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-(dX * dX / 
				(2 * std_landmark[0] * std_landmark[0]) + (dY * dY / (2 * std_landmark[1] * std_landmark[1]))));
			if (weight == 0) {
				particles[i].weight *= YAW_MIN;
			}
			else {
				particles[i].weight *= weight;
			}
			
		}
	}
	
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

	assert(is_initialized == true);

   //Uniform random number to start the wheel
	std::uniform_int_distribution<int> distInt(0, num_particles - 1);

	vector<double> weights(NUMPARTICLES);
	double maxWeight = std::numeric_limits<double>::min();
	for (int i = 0; i < num_particles; i++) {
		weights[i]= particles[i].weight;
		if (particles[i].weight > maxWeight) {
			maxWeight = particles[i].weight;
		}
	}

	//Now using the max weight I can create a distribution
	std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
	// Now we randomly select a particle, first using the distribution we select random number [0, particle number]
	int index = distInt(gen);
	double beta = 0.0;
	//Here is the wheel as explained in lessons

	/*Performance note, if we do not initialize a vector with a size, performance will suffer a lot 
	since every time is resized elements will be copied to new memory allocated */
	vector<Particle> rsParticles(NUMPARTICLES);
	for (int i = 0; i < num_particles; i++) {
		beta += distDouble(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		rsParticles[i]=(particles[index]);
	}

	particles = rsParticles;
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