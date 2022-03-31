using Convex, LinearAlgebra, Mosek, MosekTools, Random, Plots
using ScikitLearn
using StatsBase, Statistics
using DelimitedFiles
using StatsBase: sample
using ScikitLearn.CrossValidation: KFold
const MOI = Convex.MOI
@sk_import linear_model: LogisticRegression

function sdpsolver(U, w, num)
    println("Number of samples ", num);
    n = size(U)[1];
    println("Number of data ", n);
    m = size(U)[2];
    a = Variable(n);
    S = zeros(m,m);

    for i in 1:n
        data = U[i,:];
        inner = data'*w;
        matrix = data*data';
        S += a[i]*exp(inner)/(1+exp(inner))^2*matrix;
    end

    cost = 0
    for i in 1:n
        data = U[i,:];
        inner = data'*w;
        cost += exp(inner)/(1+exp(inner))^2*matrixfrac(data, S);
    end

    constraints = [0 <= a, a <= 1, sum(a) == num];
    problem = minimize(cost, constraints);
    opt = MOI.OptimizerWithAttributes(Mosek.Optimizer, "QUIET" => false)
    solve!(problem, opt);

    return a.value/num;
end

function VarianceReduction(L, Ly, U, Uy, num, X_test, y_test)
    classifier = LogisticRegression(penalty="none", max_iter=10000);
    classifier.fit(L, Ly);
    performance = zeros(2);
    pred = classifier.predict(X_test);
    performance[1] = sum(pred.==y_test)/length(y_test);
    println("Accuracy: ", performance[1]);
    w = vec(classifier.coef_);

    gamma = sdpsolver(U, w, num);
    n = size(U)[1];
    uniform = ones(n)/n;
    alpha = 1 - num^(-1/6);
    gamma = alpha*gamma+(1-alpha)*uniform;

    index = sample(1:n, Weights(vec(gamma)), num);
    data = U[index,:];
    label = Uy[index];

    classifier.fit(data, label);

    pred = classifier.predict(X_test);
    performance[2] = sum(pred.==y_test)/length(y_test);
    println("Accuracy: ", performance[2]);

    return performance
end

function experiments(name, num, initsize, k)
    X = readdlm("dataset/" * name * "_processed.dat",' ');
    m = size(X)[2]
    y = X[:,m];
    X = X[:,1:m-1];

    if k == 1
        L = X[1:initsize,:];
        Ly = y[1:initsize];
        performance = VarianceReduction(L, Ly, X, y, num, X, y);
        return performance
    else
        kf = KFold(size(X)[1], n_folds=k, shuffle=true);
        performance = zeros(k, 2);
        foldindex = 1;

        for index in kf

            train_index, test_index = index[1], index[2];

            X_train , X_test = X[train_index,:], X[test_index,:];
            y_train , y_test = y[train_index], y[test_index];

            L, Ly = X_train[1:initsize,:], y_train[1:initsize];
            U, Uy = X_train, y_train;

            performance[foldindex, :] = VarianceReduction(L, Ly, U, Uy, num, X_test, y_test);
            foldindex += 1;
        end

        return sum(performance, dims=1)/k
    end
end

function graphplotter(name, initsize, increase, ite, k)
    x = collect(0:increase:ite*increase);
    y = zeros(ite+1);
    for i in increase:increase:increse*ite
        y[i/increase+1] = experiments(name, i, initsize, k)[2];
    end
    plot(x, y, label = initsize);
    savefig(name*string(initsize));
end

graphplotter("australian", 30, 100, 6, 10);
