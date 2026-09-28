% 4C11, Lent Term 2023
% Engineering Department, University of Cambridge

% Project 1 - Problem 2: Eiffel Tower (Bars in 2D)
% Data Generation Code

clc;clear;close all
% default settings
set(0,'DefaultFigurePosition', [100 500 800 800],...
   'DefaultAxesFontSize',18,'DefaultLegendFontSize',18,...
   'DefaultLineLineWidth', 2);

nSamples = 1;
scaleFactor = 1;
yield_stress = 5e8;
Eiffel;
load_apply = zeros(nSamples, N_node);
result = zeros(nSamples, 1);

tic
for idx = 1:nSamples
    % --------------------------------------------------------------------%
    %                           PRE-PROCESSING
    % --------------------------------------------------------------------%
    % Load Eiffel Tower structure and random loading curve

    Eiffel;
    
    % --------------------------------------------------------------------%
    
    %set up FE
    nDofsPerNode        = 2;                                    % number of dof per Node
    nNodes              = size(nodalPositions,1);               % number of global nodes
    nDofs               = nNodes * nDofsPerNode;                % number of global dofs
    nElements           = size(connectivities,1);               % number of elements
    nNodesPerElement    = length(connectivities(1,1:2));        % number of nodes per element
    nDofsPerElement     = nDofsPerNode*nNodesPerElement;        % number of dofs per element
    nNeumannBCs         = size(NeumannBCs,1);                   % number of external loads applied
    nDirichletBCs       = size(DirichletBCs,1);                 % number of Dirichlet boundary conditions
    
    % elemental dofs (connectivities of the dofs)
    elementDofs = zeros(nElements,nDofsPerElement);
    
    for e = 1:nElements
        asmNode1 = connectivities(e,1:nNodesPerElement-1);
        asmNode2 = connectivities(e,nNodesPerElement);
        elementDofs(e,:) = [asmNode1*2-1, asmNode1*2,asmNode2*2-1, asmNode2*2];
    end
    
    % extract x1 and x2 for global nodes
    x1 = nodalPositions(:,1);
    x2 = nodalPositions(:,2);
    
    % let's plot our initial structure:
    if idx == 1
        hold on
           for e = 1:nElements
             asmNodes = connectivities(e,1:2);
             x1e = x1(asmNodes);
             x2e = x2(asmNodes);  
             
             xtemp = [x1e(1) x1e(2)];
             ytemp = [x2e(1) x2e(2)];
             
             p1 = plot(xtemp,ytemp,'k');
             
           end
        
        % vxlabels = arrayfun(@(n){sprintf('%d',n)},(1:nNodes)'); %numbering nodes
        % Hpl = text(x1 +0.02, x2, vxlabels,'FontWeight', 'bold','FontSize',14); %fonts etc on nodes
        p3 = plot(x1,x2,'k o', 'LineWidth',1,'MarkerEdgeColor','k',...
          'MarkerFaceColor','k');%markers on nodes
    end
    
    % --------------------------------------------------------------------%
    %                               SOLVING
    % --------------------------------------------------------------------%
    % initialise global vectors and matrices
    
    globalStiffnessMatrix           = zeros(nDofs,nDofs);
    globalForceVector               = zeros(nDofs,1);
    globalDisplacementVector        = zeros(nDofs,1);
    
    eLengthVector = zeros(nElements,1); % for post-processing purposes
    
    
    % loop over all elements 
    %-> compute elemental stiffness matrix -> assemble global stiffness matrix
    
    for e = 1:nElements
        
        asmNodes = connectivities(e,1:2);  % connectivity in terms of global nodes
        asmDofs  = elementDofs(e,:);       % connectivity in terms of global dofs
        mID      = connectivities(e,3);    % element material id
        
        x1e = x1(asmNodes);                % call known x1-positions at nodes
        x2e = x2(asmNodes);                % call known x2-positions at nodes
        
        % truss vector
        eOrientationVector = [x2e(2)-x2e(1) , x1e(2)-x1e(1)];
        
        % truss orientation
        eOrientationAngle  = atan2(eOrientationVector(1), eOrientationVector(2));
    
        %length of truss element
        eLength            = norm(eOrientationVector);                     
        eLengthVector(e)   = eLength; % save the length of each element
       
        
        % truss Area, Young's Modulus
        Ee   = mprop(mID,1);
        Ae   = mprop(mID,2);
        
        % Element Axial Stiffness Matrix coefficient   
        k    = Ae*Ee/eLength;
        
        AxialStiffnessMatrix = k* [ 1 -1
                                   -1  1];  
        % Rotation Matrix
        RotationMatrix = [cos(eOrientationAngle) sin(eOrientationAngle) 0 0
                          0 0 cos(eOrientationAngle) sin(eOrientationAngle)];           
        
        % element Stiffness Matrix in Global Coordinate System
        ElementStiffnessMatrix = RotationMatrix'*AxialStiffnessMatrix*RotationMatrix;
        
        % save stiffnesses and rotations in a struct for postprocessing
        structESMandROT(e).a = ElementStiffnessMatrix;
        structESMandROT(e).b = RotationMatrix; 
        
        %Assemble Ke into global K 
        globalStiffnessMatrix(asmDofs, asmDofs) = globalStiffnessMatrix( asmDofs,asmDofs)...
                                                           + ElementStiffnessMatrix;
                                                       
                                                       
    end
    
    % assemble global force vector
    for loadIndex=1:nNeumannBCs
    
            dof = nDofsPerNode * NeumannBCs(loadIndex,1)- nDofsPerNode + NeumannBCs(loadIndex,2);
            globalForceVector(dof) = NeumannBCs(loadIndex,3);
    
    end
    
    
    % --------------------------------------------------------------------%
    
    % make a copy of K to enforce the essential bcs 
    K = globalStiffnessMatrix;
    
    essentialBCDofs = zeros(nDirichletBCs,1)'; % if we wanted to solve by partitioning
    
    for boundIndex = 1:nDirichletBCs
    
            % Find essential boundary condition dof
            dof = nDofsPerNode * DirichletBCs(boundIndex,1)- nDofsPerNode + DirichletBCs(boundIndex,2);
       
            % Enforce essential boundary condition
            K(dof,:)   = 0;
            K(dof,dof) = 1;
            globalForceVector(dof) =  DirichletBCs(boundIndex,3);
            
    end
    
    % solve for displacement:
    globalDisplacementVector = K \ globalForceVector;
    
    % solve for reaction forces:
    globalForceVector = globalStiffnessMatrix*globalDisplacementVector;
    
    % --------------------------------------------------------------------%
    %                           POST-PROCESSING
    % --------------------------------------------------------------------%
    %calculate displaced positions
    % we introduce a scale factor because the displacements are very small
    x1_new = length(x1);
    x2_new = length(x2);
    for n = 1:nNodes
        x1_new(n) = x1(n) + scaleFactor*globalDisplacementVector(2*n-1);
        x2_new(n) = x2(n) + scaleFactor*globalDisplacementVector(2*n);
    end
    
    
    % plot displaced positions for the last sample for clearity
    if idx == nSamples
        for l = 1:nElements
             asmNodes = connectivities(l,1:2);
             x1_displaced = x1_new(asmNodes);
             x2_displaced = x2_new(asmNodes);
             p2 = plot(x1_displaced,x2_displaced,'r--');  
        end
        
        quiver(ChosenPositions(:,1),ChosenPositions(:,2),NeumannBCs(:,3)/5e5,zeros(20,1),'off','b',LineWidth=3)
        xlabel('x1')
        ylabel('x2')
        grid on
        legend([p1, p2],{'original structure','displaced structure'},'Location','northwest')
    end

    % --------------------------------------------------------------------%
    
    % compute strain, stress and energy:
    strainvector = zeros(nElements,1);
    stressvector = zeros(nElements,1);
    globalEnergy = 0;
    
    for e = 1:nElements
        mID                    = connectivities(e,3);     % element material id
        asmDofs                = elementDofs(e,:);
        elemNodalDisplacements = globalDisplacementVector(asmDofs);
        strainvector(e)        = diff((structESMandROT(e).b *elemNodalDisplacements))/eLengthVector(e); % assume small dispalcement so that the angle change of an element is negligible??? 
        stressvector(e)        = mprop(mID,1)*strainvector(e);
        globalEnergy           = globalEnergy + 0.5*(dot(elemNodalDisplacements, structESMandROT(e).a*elemNodalDisplacements));
        
    end
    
    % --------------------------------------------------------------------%
    load_apply(idx,:) = NeumannBCs(:,3).';
    if max(stressvector) > yield_stress
        result(idx,:) = 0; % if the structure fails, result = 0
    else
        result(idx,:) = 1; % if the structure holds, result = 1
    end

end
disp(mean(result))
toc

% save Eiffel_data.mat load_apply result -v7.3

