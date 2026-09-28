function rightboundary = GenBoundary(N_walk, epsi_max, time)
      N = N_walk; 
      dt_walk = randfixedsum(N,1,1,0,1);
      check = true;
      while check 
          dt_walk = randfixedsum(N,1,1,0,1);
          if min(dt_walk) > 0.02
              check = false; 
          end 
      end

    t_walk = zeros(N+1,1);

    x_t = zeros(N,1);
    x_t(1) = -epsi_max/10;

     for kkk = 1:N
      t_walk(kkk+1) = t_walk(kkk)+dt_walk(kkk);
     end

    for n = 1:N % Looping all values of N into x_t(n).
        B = rand; 
        if B < 0.5
        A = 1;
        else
        A = -1;
        end 
        x_t(n+1) = x_t(n) + epsi_max*sqrt(dt_walk(n))*A;
    end
    rightboundary = pchip(t_walk,x_t,time);
%     plot(time, rightboundary, '-');
end 