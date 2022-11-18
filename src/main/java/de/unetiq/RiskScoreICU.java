package de.unetiq;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import com.fasterxml.jackson.core.exc.StreamWriteException;
import com.fasterxml.jackson.databind.DatabindException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.HashMap;
import java.util.ArrayList;
import java.io.File;
import java.io.IOException;

/**
 * Prediction of mortality risk scores using a trained XGBoost model on ICU
 * example data.
 * 
 * @author <a href=:mailto:joceline@unetiq.com">Joceline Ziegler</a>
 */

public class RiskScoreICU {

    /**
     * Process example data from JSON files, predict risk scores for individual
     * patients and save all obtained risk scores in a JSON output file.
     * 
     */

    public String patient_data_path = "src/main/resources/data/noncovid_patient_data.json";
    public String output_path = "src/main/resources/results/noncovid_patient_results.json";
    public String model_path = "src/main/resources/models/icu-model.json";

    // Whether the scores should be written in the occurring patient order
    // If false, scores are saved in ascending order to discourage matching of
    // scores to incoming patient data
    // See DashboardDataProcessor Datensatzbeschreibung specifications
    public boolean retain_order = true;

    public static void main(String[] args) throws XGBoostError, IOException{

        RiskScoreICU risk_score_icu = new RiskScoreICU();

        // Read and process patient measurement data
        PatientDataHolder data_holder = new PatientDataHolder(risk_score_icu.patient_data_path);
        float[] data = data_holder.process_all_patients();
        DMatrix data_matrix = new DMatrix(data, data_holder.n_patients, data_holder.n_features, 0.0f);

        Booster booster = XGBoost.loadModel(risk_score_icu.model_path);
        float[][] preds = booster.predict(data_matrix);

        // Post-process predictions
        float[] all_scores = new float[data_holder.n_patients];
        for (int i = 0; i < preds.length; i++) {
            all_scores[i] = preds[i][0];
        }

        if (!risk_score_icu.retain_order) {
            Arrays.sort(all_scores);
        }

        // Save predicted risk scores to json output file
        risk_score_icu.save_prediction_results(all_scores);
    }

    private void save_prediction_results(float[] predictions)
            throws java.io.IOException, StreamWriteException, DatabindException {

        /**
         * Save predicted risk scores to a JSON file.
         * The structure follows a subpart of the specification for data items as
         * detailed in the DashboardDataProcessor Datensatzbeschreibung.
         * https://github.com/mwtek/dashboarddataprocessor/blob/main/files/Datensatzbeschreibung_COVID_dashboard_v0_3_0_final.pdf
         * 
         * @param predictions Array containing the predicted risk scores.
         */

        // Define the data item
        Map<String, Object> risk_score_item = new HashMap<String, Object>();
        risk_score_item.put("item_name", "ukr_risk_score");
        risk_score_item.put("itemtype", "aggregated");
        risk_score_item.put("data", predictions);

        // Wrap it in a list
        List<Map<String, Object>> data_items = new ArrayList<Map<String, Object>>();
        data_items.add(risk_score_item);

        // Insert it in the final data items map
        Map<String, List<Map<String, Object>>> save_scores = new HashMap<String, List<Map<String, Object>>>();
        save_scores.put("data_items", data_items);

        // Write file
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(this.output_path), save_scores);

    }
}