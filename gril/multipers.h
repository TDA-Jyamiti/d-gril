#ifndef MULTIPERS_H
#define MULTIPERS_H
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <tuple>
#include <future>
#include <map>
#include "utils.h"

#include "./phat/compute_persistence_pairs.h"

using torch::Tensor;
using namespace torch::indexing;
typedef std::pair<int, int> Point;

class Multipers{
    private:
        int hom_rank;
        std::vector<int> ranks;
        double step, ll_x, ll_y, res;
        int px, py;
        int l;
        std::vector<std::map<int, int>*> rank_info_h0;
        std::vector<std::map<int, int>*> rank_info_h1;
        int num_points_guess;
        void set_ranks(std::vector<int> ranks_){
            this->ranks.insert(this->ranks.begin(), ranks_.begin(), ranks_.end());
        }
        void set_step(double step){
            this->step = step;
        }
        void set_res(double res){
            this->res = res;
        }
        void set_l_for_worm(int l){
            this->l = l;
        }
        Tensor compute_l_worm(const int d);
        std::vector<std::tuple<bool, Integer>> compute_filtration_along_boundary_cap(const Tensor& grid_pts_along_boundary_t,
                                                                                    const Tensor& f,
                                                                                    const Tensor& f_x_sorted,
                                                                                    const Tensor& f_y_sorted,
                                                                                    const Tensor& f_x_sorted_id,
                                                                                    const Tensor& f_y_sorted_id,
                                                                                    int &manual_birth_pts,
                                                                                    int &manual_death_pts);

       void zigzag_pairs(std::vector<std::tuple<bool, Integer>> &simplices_birth_death,
                        const vector<Simplex> &simplices, 
                        const int manual_birth_pts, 
                        const int manual_death_pts,
                        std::vector<int> &num_full_bars);
        
        void num_full_bars_for_specific_d(const Tensor& filtration, 
                                            const Tensor& f_x_sorted,
                                            const Tensor& f_y_sorted,
                                            const Tensor& f_x_sorted_id,
                                            const Tensor& f_y_sorted_id,
                                            const vector<Simplex>& simplices, 
                                            const Point& p, 
                                            int d, 
                                            std::vector<int> &num_full_bars);

        Tensor find_maximal_worm_for_rank_k(const Tensor &filtration, 
                                            const Tensor& f_x_sorted,
                                            const Tensor& f_y_sorted,
                                            const Tensor& f_x_sorted_id,
                                            const Tensor& f_y_sorted_id,
                                            const vector<Simplex> &simplices, 
                                            const Point &p, 
                                            const int rank, 
                                            std::vector<std::map<int, int>*> rank_info);
        
        void set_grid_resolution_and_lower_left_corner(const Tensor& filtration);


    
    public:
        int max_threads;
        Multipers(const int hom_rank, const int l, double res, int num_centers, const std::vector<int> ranks){
            set_hom_rank(hom_rank);
            set_l_for_worm(l);
            // set_division_along_axes(px, py);
            // set_step(step);
            set_res(res);
            set_ranks(ranks);
            this->max_threads = 1;
            this->num_points_guess = num_centers;
            for(auto i = 0; i < num_points_guess; i++){
                this->rank_info_h0.push_back(new std::map<int, int>());
                this->rank_info_h1.push_back(new std::map<int, int>());
            }
        }
        void set_max_jobs(int max_jobs){
            this->max_threads = max_jobs;
        }
        std::vector<Tensor> compute_landscape(const std::vector<Point>& pts, const std::vector<std::tuple<Tensor, vector<Simplex>>> &batch);
        
        void set_hom_rank(int hom_rank){
            this->hom_rank = hom_rank;
        }
        void refresh_rank_info(){
            this->rank_info_h0.clear();
            this->rank_info_h1.clear();
            for(auto i = 0; i < num_points_guess; i++){
                this->rank_info_h0.push_back(new std::map<int, int>());
                this->rank_info_h1.push_back(new std::map<int, int>());

            }
        }
        // Tensor compute_grad_matrix(const std::vector<Point>& pts, const std::vector<std::tuple<Tensor, vector<Simplex>>> &batch);



};

#endif