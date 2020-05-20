int : nums_node;
int : nums_edge_collision;
int : nums_edge_adjacent;
float: contour_area;
set of int: NODE = 1..nums_node;
set of int: EDGE_COLL = 1..nums_edge_collision;
set of int: EDGE_ADJ = 1..nums_edge_adjacent;

array[NODE] of float : node_area;
array[NODE] of float : node_perimeter;
array[EDGE_COLL] of NODE : from_collision;
array[EDGE_COLL] of NODE : to_collision;
array[EDGE_ADJ] of NODE : from_adjacent;
array[EDGE_ADJ] of NODE : to_adjacent;
array[EDGE_ADJ] of float : align_length;

array[NODE] of var 0..1 : node;
array[EDGE_ADJ] of var 0..1 : selected_edges;

%%%%%%%%%%%%%%%%%% constraints
% constraint no collision
constraint
forall(e in EDGE_COLL) (
        node[from_collision[e]] + node[to_collision[e]] <= 1
);

% constraint all connected
% include "connected.mzn";
% constraint connected(from_adjacent, to_adjacent, node, selected_edges);

%%%%%%%%%%%%%%%%%% objectives

% calculate alignment length
var float : alignment_length_sum = sum(e in EDGE_ADJ)(
       node[from_adjacent[e]] * node[to_adjacent[e]] * align_length[e]
);

% calculate total node area
var float : node_area_obj = sum(n in NODE) (node[n] * node_area[n]) / contour_area;

% remark divide total align length
var float : alignment_obj = alignment_length_sum / sum(node_perimeter) ;

% var float : node_num = sum(node) / nums_node;

var float : obj;
obj = node_area_obj 
+ 0.0001 * alignment_obj;
solve maximize obj;


