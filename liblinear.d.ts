export interface LibLinearModule extends EmscriptenModule {
    _prepare_param(solver_type: number, C: number, eps: number, nr_weight: number, weightLabelPointer: number,
                    weightPointer: number, p: number, nu: number, initSolPointer: number, regularize_bias: number);
    
    _free_param(paramPointer: number);
    _init_problem(dataPointer: number, labelPointer: number, nr_instance: number, nr_feature: number, bias: number);
    _free_problem(problemPointer: number);
    _train_model(problemPointer: number, paramPointer: number);
    _free_model(modelPointer: number);
    _predict_one(modelPointer: number, dataPointer: number, nr_feature: number);
    _predict_one_prob(modelPointer: number, dataPointer: number);
}

export default function libsvmFactory(): Promise<LibLinearModule>;
