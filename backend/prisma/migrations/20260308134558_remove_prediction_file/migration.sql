/*
  Warnings:

  - You are about to drop the column `datasetId` on the `PredictionFile` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE "PredictionFile" DROP CONSTRAINT "PredictionFile_datasetId_fkey";

-- AlterTable
ALTER TABLE "PredictionFile" DROP COLUMN "datasetId";
