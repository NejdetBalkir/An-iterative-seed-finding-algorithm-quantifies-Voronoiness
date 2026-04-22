% Your initial data
a = [0.000; 0.001; 0.002; 0.003; 0.004; 0.005; 0.006; 0.007; 0.008; 0.009; 0.01; ...
     0.02; 0.03; 0.04; 0.05; 0.06; 0.07; 0.08; 0.09; 0.1; ...
     0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9];
     
v = [1.0000; 0.9939; 0.9888; 0.9840; 0.9795; 0.9751; 0.9709; 0.9668; 0.9629; 0.9590; 0.9552; ...
     0.9210; 0.8911; 0.8642; 0.8396; 0.8168; 0.7956; 0.7758; 0.7571; 0.7395; ...
     0.6032; 0.5112; 0.4435; 0.3902; 0.3459; 0.3078; 0.2741; 0.2441];

% 1. Isolate the "fine" data (e.g., a <= 0.01)
fine_indices = a <= 0.01;
a_fine = a(fine_indices);
v_fine = v(fine_indices);

% 2. Perform linear fit (polyfit of degree 1)
p = polyfit(a_fine, v_fine, 1);
slope = p(1);
intercept = p(2);

% 3. Generate y-values for the fit line
% Extending the line slightly beyond 0.01 so it's clearly visible
a_fit_line = linspace(0, 0.05, 100); 
v_fit_line = polyval(p, a_fit_line);

% 4. Plot original data and the linear fit
figure;
plot(a, v, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b'); 
hold on;
plot(a_fit_line, v_fit_line, 'r--', 'LineWidth', 2); 

% 5. Formatting
xlabel('a');
ylabel('v');
title('Data with Linear Fit on Fine Region');
legend('Data', sprintf('Linear Fit (Slope = %.2f)', slope));
grid on;
hold off;