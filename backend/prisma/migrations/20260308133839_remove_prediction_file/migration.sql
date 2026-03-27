/*
  Warnings:

  - You are about to drop the column `predictionId` on the `Evaluation` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE "Evaluation" DROP CONSTRAINT "Evaluation_predictionId_fkey";

-- AlterTable
ALTER TABLE "Evaluation" DROP COLUMN "predictionId";
