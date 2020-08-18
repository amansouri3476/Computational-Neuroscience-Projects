function d = which_dist_is_more_probable (pd1 , pd2 , p1 , p2 , x)
    % d = argmax_i ( p(x comes from pdi | x))
    % assumming f(x) is uniform, equivalently:
    % d = argmax_i ( f(x | x comes from pdi) * pi)
    % inputs:
    %   - pdi: probabilty dist. object (output of fitdist function)
    %   - pi: p(x comes from pdi)
    %   - x: observed outcome
    % output:
    %   - d: if d == i , it is more probabale that x comes from pdi
    % ****** @todo this function is specialized for this HW and needs to be
    % completed for general usage
    
    p1x = p1 * pdf(pd1(1) , x(1)) * pdf(pd1(2) , x(2));
    p2x = p2 * pdf(pd2(1) , x(1)) * pdf(pd2(2) , x(2));
    
    d = (p1x >= p2x) + 2*(p2x > p1x);
end