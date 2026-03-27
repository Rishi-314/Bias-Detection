/*
  Warnings:

  - Added the required column `predictionId` to the `Evaluation` table without a default value. This is not possible if the table is not empty.
  - Added the required column `reportType` to the `Evaluation` table without a default value. This is not possible if the table is not empty.

*/
-- CreateEnum
CREATE TYPE "ReportType" AS ENUM ('DEVELOPER', 'REGULATOR', 'ENDUSER', 'ALL');

-- AlterTable
ALTER TABLE "Dataset" ADD COLUMN     "targetVariable" TEXT;

-- AlterTable
ALTER TABLE "Evaluation" ADD COLUMN     "predictionId" TEXT NOT NULL,
ADD COLUMN     "reportType" "ReportType" NOT NULL;

-- CreateTable
CREATE TABLE "PredictionFile" (
    "id" TEXT NOT NULL,
    "filePath" TEXT NOT NULL,
    "uploadedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "datasetId" TEXT NOT NULL,

    CONSTRAINT "PredictionFile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SensitiveAttribute" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "evaluationId" TEXT NOT NULL,

    CONSTRAINT "SensitiveAttribute_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FairnessWeight" (
    "id" TEXT NOT NULL,
    "dimension" TEXT NOT NULL,
    "weight" INTEGER NOT NULL,
    "evaluationId" TEXT NOT NULL,

    CONSTRAINT "FairnessWeight_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "PredictionFile" ADD CONSTRAINT "PredictionFile_datasetId_fkey" FOREIGN KEY ("datasetId") REFERENCES "Dataset"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SensitiveAttribute" ADD CONSTRAINT "SensitiveAttribute_evaluationId_fkey" FOREIGN KEY ("evaluationId") REFERENCES "Evaluation"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "FairnessWeight" ADD CONSTRAINT "FairnessWeight_evaluationId_fkey" FOREIGN KEY ("evaluationId") REFERENCES "Evaluation"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Evaluation" ADD CONSTRAINT "Evaluation_predictionId_fkey" FOREIGN KEY ("predictionId") REFERENCES "PredictionFile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
