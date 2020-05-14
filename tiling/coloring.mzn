int : nums_node;
int : nums_edge_adjacent;
set of int: NODE = 0..nums_node-1;
set of int: COLOR = 0..3;
float : color_max_ratio = 0.30;
float : color_min_ratio = 0.20;
set of int: EDGE_ADJ = 1..nums_edge_adjacent;

array[NODE, COLOR] of var 0..1 : node_color;

array[EDGE_ADJ] of NODE : from_adjacent;
array[EDGE_ADJ] of NODE : to_adjacent;

constraint forall(e in EDGE_ADJ)(
  forall(c in COLOR) (
    node_color[from_adjacent[e], c] + node_color[to_adjacent[e], c] <=1
  )
);

constraint forall(n in NODE) (
  sum(c in COLOR) (node_color[n,c]) = 1
);

% constraint forall (c in COLOR) (
%   let {var int : color_cnt = sum(n in NODE)(node_color[n,c]);} in
%   color_cnt > color_min_ratio * nums_node /\ color_cnt < color_max_ratio * nums_node
% );

var float : obj = sum (c in COLOR) (
  let {var int : color_cnt = sum(n in NODE)(node_color[n,c]);} in
  abs(color_cnt - color_min_ratio * nums_node)
);
solve minimize obj;